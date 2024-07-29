#!/bin/bash
#SBATCH --partition=allcpu              # Node Partion(s)
#SBATCH --time=12:00:00                        # Maximum time requested
#SBATCH --nodes=1                              # Number of nodes
#SBATCH --chdir=/home/samonaco/ANNNI2/slurm/   # directory must already exist!
#SBATCH --job-name=ANNNISTATES                 # Job name
#SBATCH --output=fid.out            # File to which STDOUT will be written
#SBATCH --error=fid.err           # File to which STDERR will be written
#SBATCH --mail-type=END                        # Type of email notification- BEGIN,END,FAIL,ALL


echo "Activating the environment"
source /home/samonaco/ANNNI2/env/bin/activate 

echo "Running the script"
bash /home/samonaco/ANNNI2/src/ANNNI/scripts/dmrg/study3.sh 

# Write a message to the error log
>&2 echo "This is the error log message."


