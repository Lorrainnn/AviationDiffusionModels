#!/bin/bash
#SBATCH --job-name=eval_newsplit
#SBATCH --account=fc_mixedav
#SBATCH --partition=savio3_gpu
#SBATCH --qos=gtx2080_gpu3_normal
#SBATCH --gres=gpu:GTX2080TI:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/eval_%A_%a.out
#SBATCH --error=logs/eval_%A_%a.err

# one array task per model, each on its own GPU, each does K=5 first and then K=20.
# run `mkdir -p logs` BEFORE sbatch.
MODELS=(CFM-NewSplit RoPE-OG RoPE-TS RoPE-TS-Impose)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

module load anaconda3
source activate adsb

python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print('Device:', torch.cuda.get_device_name(0))" || { echo "CUDA check failed, aborting"; exit 1; }

NC_PATH=/global/scratch/users/kishore26/adsb-diffusion/data/trajectories_adsblol_seq86_stage2.nc
OUT=$HOME/eval_results_newSplit
mkdir -p $OUT

cd $HOME/AviationDiffusionModelsKishore/training

echo "Evaluating $MODEL"
python -u evalAblationSavio.py \
    --nc_path $NC_PATH \
    --models $MODEL \
    --Ks 5,20 \
    --ckpt_root $HOME \
    --batch_size 128 \
    --out $OUT

echo "Job finished at $(date)"