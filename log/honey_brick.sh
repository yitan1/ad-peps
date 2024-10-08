#!/bin/bash
#SBATCH --job-name exci
#SBATCH --partition 256G56c
#SBATCH --nodes=1  
#SBATCH --exclusive   #独占节点 Exclusive execution mode
#SBATCH --ntasks-per-node=56   #number of cores

module load anaconda

# Compute ground-state
dir_name="honey_brick_11"
mkdir ${dir_name}
cd ..
python -m adpeps gs honey_brick.yaml > log/${dir_name}/gs_honey_brick_$SLURM_JOBID.log 2>&1

# Construct exci basis
python -m adpeps exci honey_brick_exci.yaml -i > log/${dir_name}/basis_honey_brick_$SLURM_JOBID.log 2>&1

# Compute exci along k-space path
for((i=1;i<=31;i++));  
do   
python -m adpeps exci honey_brick_exci.yaml -p $i > log/${dir_name}/exci_honey_brick_${i}_$SLURM_JOBID.log 2>&1 ;
done  