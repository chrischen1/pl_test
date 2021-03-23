# Begin LSF Directives
#BSUB -P bif132
#BSUB -W 2:00
#BSUB -nnodes 1
#BSUB -J test
#BSUB -o test.%J
#BSUB -e test.%J
#BSUB -alloc_flags "gpumps"


export DGLBACKEND=pytorch  # Required to override default ~/.dgl config directory which is read-only
export WANDB_CONFIG_DIR=/gpfs/alpine/scratch/chen08900/bif132/  # For local reading and writing of WandB files
export WANDB_CACHE_DIR=/gpfs/alpine/scratch/chen08900/bif132/  # For logging checkpoints as artifacts
export all_proxy=socks://proxy.ccs.ornl.gov:3128/
export ftp_proxy=ftp://proxy.ccs.ornl.gov:3128/
export http_proxy=http://proxy.ccs.ornl.gov:3128/
export https_proxy=https://proxy.ccs.ornl.gov:3128/
export no_proxy='localhost,127.0.0.0/8,.ccs.ornl.gov,.ncrc.gov'

module load gcc/10.2.0
module load open-ce
module load cuda/10.2.89
export OMP_NUM_THREADS=4
conda activate torch_gpu

date
jsrun --smpiargs="off"  -g6 -a1 -c4 -r1 python train.py --num_gpus 6 --test_size 500 --epochs 50 --out_path test_node2 --batch_size 32
date