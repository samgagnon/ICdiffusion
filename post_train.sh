#!/bin/bash
#SBATCH -A IscrC_ION-DIFF
#SBATCH -p boost_usr_prod
#SBATCH --time 24:00:00     # format: HH:MM:SS
#SBATCH -N 1                # 1 node
#SBATCH --ntasks-per-node=2 # 4 tasks out of 32
#SBATCH --gres=gpu:2        # 4 gpus per node out of 4
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

python post_train.py -j galsmear --num_workers 2 --num_bins 1 --nf 0.4
python post_train.py -j galsmear --num_workers 2 --num_bins 2 --nf 0.4
python post_train.py -j galsmear --num_workers 2 --num_bins 3 --nf 0.4