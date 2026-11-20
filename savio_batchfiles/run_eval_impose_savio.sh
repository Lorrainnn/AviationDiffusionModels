#!/bin/bash
#SBATCH --job-name=eval_impose
#SBATCH --account=fc_mixedav
#SBATCH --partition=savio3_gpu
#SBATCH --qos=gtx2080_gpu3_normal
#SBATCH --gres=gpu:GTX2080TI:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=12:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/eval_impose_%A_%a.out
#SBATCH --error=logs/eval_impose_%A_%a.err

# task 0 -> K=5, task 1 -> K=20, in parallel on two GPUs, so K=5 lands first.
# run `mkdir -p logs` in the folder you sbatch from BEFORE sbatch.
KS=(5 20)
K=${KS[$SLURM_ARRAY_TASK_ID]}

module load anaconda3
source activate adsb

python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print('Device:', torch.cuda.get_device_name(0))" || { echo "CUDA check failed, aborting"; exit 1; }

NC_PATH=/global/scratch/users/kishore26/adsb-diffusion/data/trajectories_adsblol_seq86_stage2.nc
OUT=$HOME/eval_results_newSplit
mkdir -p $OUT

cd $HOME/AviationDiffusionModelsKishore/training

echo "Imposition eval, K=$K, k=0,1,2,3"
python -u evalImposeSavio.py \
    --nc_path $NC_PATH \
    --Ks $K \
    --ks 0,1,2,3 \
    --ckpt_root $HOME \
    --batch_size 128 \
    --out $OUT

echo "Job finished at $(date)"