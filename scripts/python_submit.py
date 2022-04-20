import os
import time

#--nodelist=roma

def python_submit(command, node = None):
    bash_file = open("./slurm.sh","w")
    bash_file.write(f'#!/bin/bash\n{command}')
    bash_file.close()
    if node == None:
        os.system('sbatch -c 8 --gres=gpu:1 --time=1-00:00:00 slurm.sh')
    else:
        os.system(f'sbatch -c 8 --gres=gpu:1 --nodelist={node} --time=1-00:00:00 slurm.sh')
    os.remove("./slurm.sh")


