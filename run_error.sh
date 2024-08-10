#!/bin/bash
#SBATCH --mem=64GB
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=4-00:00:00

MODELS1=(
    "/mnt/parscratch/users/lip23ss/models/Q-bert/Mamba-370M_magnitude_0.1"
    "/mnt/parscratch/users/lip23ss/models/Q-bert/Mamba-370M_magnitude_0.2"
    "/mnt/parscratch/users/lip23ss/models/Q-bert/Mamba-370M_magnitude_0.3"
    "/mnt/parscratch/users/lip23ss/models/Q-bert/Mamba-370M_magnitude_0.4"
    "/mnt/parscratch/users/lip23ss/models/Q-bert/Mamba-370M_magnitude_0.5"
    "/mnt/parscratch/users/lip23ss/models/Q-bert/Mamba-370M_sparsegpt_0.1"
    "/mnt/parscratch/users/lip23ss/models/Q-bert/Mamba-370M_sparsegpt_0.2"
    "/mnt/parscratch/users/lip23ss/models/Q-bert/Mamba-370M_sparsegpt_0.3"
    "/mnt/parscratch/users/lip23ss/models/Q-bert/Mamba-370M_sparsegpt_0.4"
    "/mnt/parscratch/users/lip23ss/models/Q-bert/Mamba-370M_sparsegpt_0.5"
    "/mnt/parscratch/users/lip23ss/models/Q-bert/Mamba-370M_wanda_0.1"
    "/mnt/parscratch/users/lip23ss/models/Q-bert/Mamba-370M_wanda_0.2"
    "/mnt/parscratch/users/lip23ss/models/Q-bert/Mamba-370M_wanda_0.3"
    "/mnt/parscratch/users/lip23ss/models/Q-bert/Mamba-370M_wanda_0.4"
    "/mnt/parscratch/users/lip23ss/models/Q-bert/Mamba-370M_wanda_0.5"
)

MODELS2=(
    "/mnt/parscratch/users/lip23ss/models/Q-bert/Mamba-1B_magnitude_0.1"
    "/mnt/parscratch/users/lip23ss/models/Q-bert/Mamba-1B_magnitude_0.2"
    "/mnt/parscratch/users/lip23ss/models/Q-bert/Mamba-1B_magnitude_0.3"
    "/mnt/parscratch/users/lip23ss/models/Q-bert/Mamba-1B_magnitude_0.4"
    "/mnt/parscratch/users/lip23ss/models/Q-bert/Mamba-1B_magnitude_0.5"
    "/mnt/parscratch/users/lip23ss/models/Q-bert/Mamba-1B_sparsegpt_0.1"
    "/mnt/parscratch/users/lip23ss/models/Q-bert/Mamba-1B_sparsegpt_0.2"
    "/mnt/parscratch/users/lip23ss/models/Q-bert/Mamba-1B_sparsegpt_0.3"
    "/mnt/parscratch/users/lip23ss/models/Q-bert/Mamba-1B_sparsegpt_0.4"
    "/mnt/parscratch/users/lip23ss/models/Q-bert/Mamba-1B_sparsegpt_0.5"
    "/mnt/parscratch/users/lip23ss/models/Q-bert/Mamba-1B_wanda_0.1"
    "/mnt/parscratch/users/lip23ss/models/Q-bert/Mamba-1B_wanda_0.2"
    "/mnt/parscratch/users/lip23ss/models/Q-bert/Mamba-1B_wanda_0.3"
    "/mnt/parscratch/users/lip23ss/models/Q-bert/Mamba-1B_wanda_0.4"
    "/mnt/parscratch/users/lip23ss/models/Q-bert/Mamba-1B_wanda_0.5"
)

MODELS3=(
    "/mnt/parscratch/users/lip23ss/models/Q-bert/Mamba-3B_magnitude_0.1"
    "/mnt/parscratch/users/lip23ss/models/Q-bert/Mamba-3B_magnitude_0.2"
    "/mnt/parscratch/users/lip23ss/models/Q-bert/Mamba-3B_magnitude_0.3"
    "/mnt/parscratch/users/lip23ss/models/Q-bert/Mamba-3B_magnitude_0.4"
    "/mnt/parscratch/users/lip23ss/models/Q-bert/Mamba-3B_magnitude_0.5"
    "/mnt/parscratch/users/lip23ss/models/Q-bert/Mamba-3B_sparsegpt_0.1"
    "/mnt/parscratch/users/lip23ss/models/Q-bert/Mamba-3B_sparsegpt_0.2"
    "/mnt/parscratch/users/lip23ss/models/Q-bert/Mamba-3B_sparsegpt_0.3"
    "/mnt/parscratch/users/lip23ss/models/Q-bert/Mamba-3B_sparsegpt_0.4"
    "/mnt/parscratch/users/lip23ss/models/Q-bert/Mamba-3B_sparsegpt_0.5"
    "/mnt/parscratch/users/lip23ss/models/Q-bert/Mamba-3B_wanda_0.1"
    "/mnt/parscratch/users/lip23ss/models/Q-bert/Mamba-3B_wanda_0.2"
    "/mnt/parscratch/users/lip23ss/models/Q-bert/Mamba-3B_wanda_0.3"
    "/mnt/parscratch/users/lip23ss/models/Q-bert/Mamba-3B_wanda_0.4"
    "/mnt/parscratch/users/lip23ss/models/Q-bert/Mamba-3B_wanda_0.5"
)

for MODEL in "${MODELS1[@]}"; do
    JOB_ID=$(sbatch --parsable --export=MODEL=$MODEL,FULLMODEL="Q-bert/Mamba-370M" error.sh)
done

for MODEL in "${MODELS2[@]}"; do
    JOB_ID=$(sbatch --parsable --export=MODEL=$MODEL,FULLMODEL="Q-bert/Mamba-1B" error.sh)
done

for MODEL in "${MODELS3[@]}"; do
    JOB_ID=$(sbatch --parsable --export=MODEL=$MODEL,FULLMODEL="Q-bert/Mamba-3B" error.sh)
done