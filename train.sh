#!/bin/bash
#SBATCH -A IscrC_ION-DIFF
#SBATCH -p boost_usr_prod
#SBATCH --time 24:00:00     # format: HH:MM:SS
#SBATCH -N 1                # 1 node
#SBATCH --ntasks-per-node=4 # 4 tasks out of 32
#SBATCH --gres=gpu:4        # 4 gpus per node out of 4
#SBATCH --mem=63000         # memory per node out of 494000MB
#SBATCH --job-name=train_network
#SBATCH --mail-user=samuel.gagnonhartman@sns.it
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL


module load profile/deeplrn
source ~/ddpm/bin/activate

python train.py -j galsmear --num_workers 4 --num_bins 1
python sample.py -j galsmear --num_workers 4 --num_bins 1 --task_id training
python sample.py -j galsmear --num_workers 4 --num_bins 1 --task_id validation

scp -r /scratch/* ./results/
