import torch
import torch.nn as nn
import torchvision
from torch.nn import Module
import urllib
import pathlib
import torch.nn.functional as F
import os
import numpy as np

model_urls = {}

model_urls['101-elastic'] = {
    "description": 'This model was trained on the datasets: nih pc rsna mimic_ch chex datasets.',
    "weights_url": 'https://github.com/mlmed/torchxrayvision/releases/download/v1/nihpcrsnamimic_ch-resnet101-2-ae-test2-elastic-e250.pt',
    "image_range": [-1024,1024],
    "class":"ResNetAE101"
    }

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, progress=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample

    def forward(self, x):
        shortcut = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        if self.downsample is not None:
            shortcut = self.downsample(x)

        out = out + shortcut
        out = self.relu(out)

        return out


class DeconvBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=2, stride=1, upsample=None):
        super(DeconvBottleneck, self).__init__()
        self.expansion = expansion
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        if stride == 1:
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                   stride=stride, bias=False, padding=1)
        else:
            self.conv2 = nn.ConvTranspose2d(out_channels, out_channels,
                                            kernel_size=3,
                                            stride=stride, bias=False,
                                            padding=1,
                                            output_padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.upsample = upsample

    def forward(self, x):
        shortcut = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        if self.upsample is not None:
            shortcut = self.upsample(x)

        out = out + shortcut
        out = self.relu(out)

        return out

# source: https://github.com/ycszen/pytorch-segmentation/blob/master/resnet.py
class _ResNetAE(nn.Module):
    def __init__(self, downblock, upblock, num_layers, n_classes):
        super(_ResNetAE, self).__init__()
        
        self.in_channels = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_downlayer(downblock, 64, num_layers[0])
        self.layer2 = self._make_downlayer(downblock, 128, num_layers[1], stride=2)
        self.layer3 = self._make_downlayer(downblock, 256, num_layers[2], stride=2)
        self.layer4 = self._make_downlayer(downblock, 128, num_layers[3], stride=6)

        self.uplayer1 = self._make_up_block(upblock, 128, num_layers[3], stride=6)
        self.uplayer2 = self._make_up_block(upblock, 64, num_layers[2], stride=2)
        self.uplayer3 = self._make_up_block(upblock, 32, num_layers[1], stride=2)
        self.uplayer4 = self._make_up_block(upblock, 16, num_layers[0], stride=2)

        upsample = nn.Sequential(
            nn.ConvTranspose2d(self.in_channels, 64, kernel_size=1, stride=2, bias=False, output_padding=1),
            nn.BatchNorm2d(64),
        )
        self.uplayer_top = DeconvBottleneck(self.in_channels, 64, 1, 2, upsample)

        self.conv1_1 = nn.ConvTranspose2d(64, n_classes, kernel_size=1, stride=1, bias=False)
        
    def __repr__(self):
        if self.weights != None:
            return "XRV-ResNetAE-{}".format(self.weights)
        else:
            return "XRV-ResNetAE"
        
    def _make_downlayer(self, block, init_channels, num_layer, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != init_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, init_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(init_channels * block.expansion),
            )
        layers = []
        layers.append(block(self.in_channels, init_channels, stride, downsample))
        self.in_channels = init_channels * block.expansion
        for i in range(1, num_layer):
            layers.append(block(self.in_channels, init_channels))

        return nn.Sequential(*layers)

    def _make_up_block(self, block, init_channels, num_layer, stride=1):
        upsample = None
        # expansion = block.expansion
        if stride != 1 or self.in_channels != init_channels * 2:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.in_channels, init_channels * 2,kernel_size=1, stride=stride, bias=False, output_padding=1),
                nn.BatchNorm2d(init_channels * 2),
            )
        layers = []
        for i in range(1, num_layer):
            layers.append(block(self.in_channels, init_channels, 4))
            
        layers.append(block(self.in_channels, init_channels, 2, stride, upsample))
        self.in_channels = init_channels * 2
        return nn.Sequential(*layers)

    def encode(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
    
    def features(self, x):
        return self.encode(x)

    def decode(self, x, image_size=[1,1,224,224]):
        x = self.uplayer1(x)
        x = self.uplayer2(x)
        x = self.uplayer3(x)
        x = self.uplayer4(x)
        x = self.uplayer_top(x)

        x = self.conv1_1(x, output_size=image_size)
        x = nn.Sigmoid()(x)
        return x
    
    def forward(self, x):
        ret = {}
        ret["z"] = z = self.encode(x)
        #print(z.shape)
        ret["out"] = self.decode(z, x.size())

        return ret
    
def ResNetAE50(**kwargs):
    return _ResNetAE(Bottleneck, DeconvBottleneck, [3, 4, 6, 3], 1, **kwargs)


def ResNetAE101(**kwargs):
    return _ResNetAE(Bottleneck, DeconvBottleneck, [3, 4, 23, 2], 1, **kwargs)


def ResNetAE(weights=None):
    
    if weights == None:
        return ResNetAE101()
    
    if not weights in model_urls.keys():
        raise Exception("weights value must be in {}".format(list(model_urls.keys())))
    
    method_to_call = globals()[model_urls[weights]["class"]]
    ae = method_to_call()
    
    ## load pretrained models
    url = model_urls[weights]["weights_url"]
    weights_filename = os.path.basename(url)
    weights_storage_folder = os.path.expanduser(os.path.join("~",".torchxrayvision","models_data"))
    weights_filename_local = os.path.expanduser(os.path.join(weights_storage_folder,weights_filename))

    if not os.path.isfile(weights_filename_local):
        print("Downloading weights...")
        print("If this fails you can run `wget {} -O {}`".format(url, weights_filename_local))
        pathlib.Path(weights_storage_folder).mkdir(parents=True, exist_ok=True)
        download(url, weights_filename_local)

    try:
        state_dict = torch.load(weights_filename_local, map_location='cpu')
        ae.load_state_dict(state_dict)
    except Exception as e:
        print("Loading failure. Check weights file:", weights_filename_local)
        raise(e)
        
    ae = ae.eval()
    
    ae.weights = weights
    ae.description = model_urls[weights]["description"]
    
    return ae
    
import sys
import requests

# from here https://sumit-ghosh.com/articles/python-download-progress-bar/
def download(url, filename):
    with open(filename, 'wb') as f:
        response = requests.get(url, stream=True)
        total = response.headers.get('content-length')

        if total is None:
            f.write(response.content)
        else:
            downloaded = 0
            total = int(total)
            for data in response.iter_content(chunk_size=max(int(total/1000), 1024*1024)):
                downloaded += len(data)
                f.write(data)
                done = int(50*downloaded/total)
                sys.stdout.write('\r[{}{}]'.format('█' * done, '.' * (50-done)))
                sys.stdout.flush()
    sys.stdout.write('\n')


import torch
from .base import BaseVAE
from torch import nn
from torchvision.models import vgg19_bn
from torch.nn import functional as F
from .types_ import *

class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()
    
    def forward(self, x):
        # Do your print / debug stuff here
        print(x.shape)
        return x

class DFCVAE(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 alpha:float = 1,
                 beta:float = 0.5,
                 **kwargs) -> None:
        super(DFCVAE, self).__init__()

        self.latent_dim = latent_dim
        self.alpha = alpha
        self.beta = beta

        modules = []
        if hidden_dims is None:
            #hidden_dims = [32, 64, 128, 256, 512]
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]* 256, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]* 256, latent_dim)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] *  256)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 1,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())

        self.feature_network = vgg19_bn(pretrained=True)

        # Freeze the pretrained feature network
        for param in self.feature_network.parameters():
            param.requires_grad = False

        self.feature_network.eval()


    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        
        #Commenting out for now
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        
        #Commenting out for now
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]
        #return result

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 16, 16)
        result = self.decoder(result)
        #result = self.decoder(z)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        #print("INPUT")
        #print(input.shape)
        #print(z.shape)
        
        mu, log_var = self.encode(input)
        #z = self.encode(input)

        z = self.reparameterize(mu, log_var)

        #print("Z shape:")
        #print(z.shape)
        recons = self.decode(z)
        #print("Recons shape:")
        #print(recons.shape)

        #recons_features = self.extract_features(recons)
        #input_features = self.extract_features(input)
        ret = {}
        ret["z"] = z 
        #print(z.shape)
        ret["out"] = recons
        #return  [recons, input, recons_features, input_features, mu, log_var]
        return ret

    def extract_features(self,
                         input: Tensor,
                         feature_layers: List = None) -> List[Tensor]:
        """
        Extracts the features from the pretrained model
        at the layers indicated by feature_layers.
        :param input: (Tensor) [B x C x H x W]
        :param feature_layers: List of string of IDs
        :return: List of the extracted features
        """
        if feature_layers is None:
            feature_layers = ['14', '24', '34', '43']
        features = []
        result = input
        for (key, module) in self.feature_network.features._modules.items():
            result = module(result)
            if(key in feature_layers):
                features.append(result)

        return features

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        recons_features = args[2]
        input_features = args[3]
        mu = args[4]
        log_var = args[5]

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, input)

        feature_loss = 0.0
        for (r, i) in zip(recons_features, input_features):
            feature_loss += F.mse_loss(r, i)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = self.beta * (recons_loss + feature_loss) + self.alpha * kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]


class SimpleAE(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 alpha:float = 1,
                 beta:float = 0.5,
                 **kwargs) -> None:
        super(SimpleAE, self).__init__()

        self.latent_dim = latent_dim
        self.alpha = alpha
        self.beta = beta

        modules = []
        if hidden_dims is None:
            #hidden_dims = [32, 64, 128, 256, 512]
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]* 256, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]* 256, latent_dim)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] *  256)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 1,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())

        self.feature_network = vgg19_bn(pretrained=True)

        # Freeze the pretrained feature network
        for param in self.feature_network.parameters():
            param.requires_grad = False

        self.feature_network.eval()


    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        
        #Commenting out for now
        #result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        
        #Commenting out for now
        #mu = self.fc_mu(result)
        #log_var = self.fc_var(result)

        #return [mu, log_var]
        return result

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        #result = self.decoder_input(z)
        #result = result.view(-1, 512, 16, 16)
        #result = self.decoder(result)
        result = self.decoder(z)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        #print("INPUT")
        #print(input.shape)
        #print(z.shape)
        
        #mu, log_var = self.encode(input)
        z = self.encode(input)

        #z = self.reparameterize(mu, log_var)

        #print("Z shape:")
        #print(z.shape)
        recons = self.decode(z)
        #print("Recons shape:")
        #print(recons.shape)

        #recons_features = self.extract_features(recons)
        #input_features = self.extract_features(input)
        ret = {}
        ret["z"] = z 
        #print(z.shape)
        ret["out"] = recons
        #return  [recons, input, recons_features, input_features, mu, log_var]
        return ret

    def extract_features(self,
                         input: Tensor,
                         feature_layers: List = None) -> List[Tensor]:
        """
        Extracts the features from the pretrained model
        at the layers indicated by feature_layers.
        :param input: (Tensor) [B x C x H x W]
        :param feature_layers: List of string of IDs
        :return: List of the extracted features
        """
        if feature_layers is None:
            feature_layers = ['14', '24', '34', '43']
        features = []
        result = input
        for (key, module) in self.feature_network.features._modules.items():
            result = module(result)
            if(key in feature_layers):
                features.append(result)

        return features

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        recons_features = args[2]
        input_features = args[3]
        mu = args[4]
        log_var = args[5]

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, input)

        feature_loss = 0.0
        for (r, i) in zip(recons_features, input_features):
            feature_loss += F.mse_loss(r, i)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = self.beta * (recons_loss + feature_loss) + self.alpha * kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
    