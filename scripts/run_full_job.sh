#!/bin/bash
#SBATCH --time=18:00:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:4
#SBATCH --job-name=unet-segmentation
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=514404
#SBATCH --signal=B:12@600

module purge
module load Python/3.13.1-GCCcore-14.2.0
source $HOME/venvs/sem-seg/bin/activate

cp data.tar.gz $TMPDIR/data.tar.gz
tar xzf $TMPDIR/data.tar.gz -C $TMPDIR/

trap 'DEST=/scratch/$USER/u-net/job_${SLURM_JOBID}; mkdir -p $DEST; find $TMPDIR/outputs -name "*_best.pth" -exec cp --parents {} $DEST \;; find $TMPDIR/outputs -name "*.log" -exec cp --parents {} $DEST \;; find $TMPDIR/outputs -name "cfg.yaml" -exec cp --parents {} $DEST \;' 12
(
    python3 -m scripts.run_hyperparam_tuning --model unet --loss weighted_cle
    python3 -m scripts.train_ddp --model unet --loss weighted_cle
    python3 -m scripts.eval --model unet --run-id 1
    python3 -m scripts.train_ddp --model unet --loss OHEMLoss
    python3 -m scripts.eval --model unet --run-id 2
    python3 -m scripts.train_ddp --model unet --loss mixed_cle_dice
    python3 -m scripts.eval --model unet --run-id 3
    python3 -m scripts.train_ddp --model unet --loss dice
    python3 -m scripts.eval --model unet --run-id 4

    python3 -m scripts.train_ddp --model unet --loss best
    python3 -m scripts.eval --model unet --run-id 5
    python3 -m scripts.train_ddp --model unet --loss best
    python3 -m scripts.eval --model unet --run-id 6
    python3 -m scripts.train_ddp --model unet --loss best
    python3 -m scripts.eval --model unet --run-id 7
    python3 -m scripts.train_ddp --model unet --loss best
    python3 -m scripts.eval --model unet --run-id 8
) & wait


# copy things back
DEST=/scratch/$USER/u-net/job_${SLURM_JOBID}
mkdir -p $DEST
find $TMPDIR/outputs -name "*_best.pth" -exec cp --parents {} $DEST \;
find $TMPDIR/outputs -name "*.log" -exec cp --parents {} $DEST \;
find $TMPDIR/outputs -name "cfg.yaml" -exec cp --parents {} $DEST \;