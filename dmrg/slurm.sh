#!/bin/bash
#SBATCH --partition=allgpu,maxgpu              # Node Partion(s)
#SBATCH --constraint='GPUx1'                   # optional constraints (here we request one gpu)
#SBATCH --time=12:00:00                        # Maximum time requested
#SBATCH --nodes=1                              # Number of nodes
#SBATCH --chdir=/home/samonaco/ANNNI2/slurm/   # directory must already exist!
#SBATCH --job-name=ANNNISTATES                 # Job name
#SBATCH --output=ANNNISTATES_%j.out            # File to which STDOUT will be written
#SBATCH --error=ANNNISTATES_%_%j.err           # File to which STDERR will be written
#SBATCH --mail-type=END                        # Type of email notification- BEGIN,END,FAIL,ALL


echo "Activating the environment"
source ../env/dmrg/bin/activate 

echo "Running the script"
bash study.sh 

# Write a message to the error log
>&2 echo "This is the error log message."

