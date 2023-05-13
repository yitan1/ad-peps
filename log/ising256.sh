#!/bin/bash
#SBATCH --job-name exci
#SBATCH --partition 256G56c
#SBATCH --nodes=1  
#SBATCH --exclusive   #独占节点 Exclusive execution mode
#SBATCH --ntasks-per-node=56   #number of cores

dir_name="ising_h2.5"
mkdir ${dir_name}
cd ..
module load anaconda

# Compute ground-state
python -m adpeps gs ising_D2.yaml > log/${dir_name}/gs_ising_$SLURM_JOBID.log 2>&1

# Construct exci basis
python -m adpeps exci ising_D2_exci.yaml -i > log/${dir_name}/basis_ising_$SLURM_JOBID.log 2>&1

# Compute exci along k-space path
for((i=1;i<=31;i++));  
do   
python -m adpeps exci ising_D2_exci.yaml -p $i > log/${dir_name}/exci_ising_${i}_$SLURM_JOBID.log 2>&1 ;
done  