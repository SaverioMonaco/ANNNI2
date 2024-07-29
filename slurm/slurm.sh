#!/bin/bash
#SBATCH --partition=allcpu              # Node Partion(s)
#SBATCH --time=24:00:00                        # Maximum time requested
#SBATCH --nodes=1                              # Number of nodes
#SBATCH --chdir=/home/samonaco/ANNNI2/slurm/   # directory must already exist!
#SBATCH --job-name=ANNNISTATES                 # Job name
#SBATCH --output=out.log            # File to which STDOUT will be written
#SBATCH --error=err.log           # File to which STDERR will be written
#SBATCH --mail-type=END                        # Type of email notification- BEGIN,END,FAIL,ALL


echo "Activating the environment"
source ../env/bin/activate 

echo "Running the script"
bash ../src/ANNNI/scripts/dmrg/study.sh 

# Write a message to the error log
>&2 echo "This is the error log message."

