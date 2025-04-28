#!/bin/bash
#SBATCH -A IscrC_ION-DIFF
#SBATCH -p boost_usr_prod
#SBATCH --time 10:00:00     # format: HH:MM:SS
#SBATCH -N 1                # 1 node
#SBATCH --ntasks-per-node=4 # 4 tasks out of 32
#SBATCH --gres=gpu:4        # 4 gpus per node out of 4
#SBATCH --mem=63000         # memory per node out of 494000MB
#SBATCH --array=0-6750
#SBATCH --job-name=train_network
#SBATCH --mail-user=samuel.gagnonhartman@sns.it
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL


module load profile/deeplrn
source ~/ddpm/bin/activate

python hyperparams.py -j galsmear --num_workers 4 --num_bins 1 --hp_index $SLURM_ARRAY_TASK_ID