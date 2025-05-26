#!/bin/bash
#SBATCH --time=05:00:00
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

trap 'DEST=/scratch/$USER/u-net/job_${SLURM_JOBID}; mkdir -p $DEST; tar czvf $DEST/outputs_timeout.tar.gz $TMPDIR/outputs' 12
(
    python3 -m scripts.find_batch_size --model unet
    python3 -m scripts.train_ddp --model unet
    python3 -m scripts.eval --model unet --run-id 1
) & wait


# copy things back
DEST=/scratch/$USER/u-net/job_${SLURM_JOBID}
mkdir -p $DEST
cp -r $TMPDIR/outputs $DEST/