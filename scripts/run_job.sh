#!/bin/bash
#SBATCH --time=07:00:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:4
#SBATCH --job-name=unet-segmentation
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=514404

module purge
module load Python/3.13.1-GCCcore-14.2.0
cp -r $HOME/venvs/sem-seg $TMPDIR/
source $TMPDIR/sem-seg/bin/activate

cp -r $HOME/jobs/U-Net-SSeg/data $TMPDIR/
cd $HOME/jobs/U-Net-SSeg

python3 -m scripts.find_batch_size --model unet
rm -rf $TMPDIR/outputs/unet
python3 -m scripts.train_ddp --model unet
python3 -m scripts.eval --model unet --run 1




# copy things back
mkdir -p /scratch/$USER/jobs/job_${SLURM_JOBID}
cp -r $TMPDIR/outputs /scratch/$USER/jobs/job_${SLURM_JOBID}