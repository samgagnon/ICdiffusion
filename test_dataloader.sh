#!/bin/bash
#SBATCH -A IscrC_ION-DIFF
#SBATCH -p boost_usr_prod
#SBATCH --time 03:00:00     # format: HH:MM:SS
#SBATCH -N 1                # 1 node
#SBATCH --ntasks-per-node=2 # 4 tasks out of 32
#SBATCH --gres=gpu:2        # 4 gpus per node out of 4
#SBATCH --mem=63000         # memory per node out of 494000MB
#SBATCH --job-name=test_dataloader
#SBATCH --mail-user=samuel.gagnonhartman@sns.it
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL


module load profile/deeplrn
source ~/ddpm/bin/activate

# export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'

# python dataloader.py --num_workers 1 --num_bins 1
python dataloader.py --num_workers 2 --num_bins 1
python dataloader.py --num_workers 2 --num_bins 2
python dataloader.py --num_workers 2 --num_bins 3
# python dataloader.py --num_workers 3 --num_bins 1
# python dataloader.py --num_workers 4 --num_bins 1
