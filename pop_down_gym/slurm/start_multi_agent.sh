#!/bin/bash
sbatch <<EOT
#!/bin/bash

#SBATCH --ntasks=2
#SBATCH --cpus-per-task=16
#SBATCH --partition=sched_mit_psfc_gpu_r8
#SBATCH --time=0-04:00:0
#SBATCH --output=/home/allenw/Scratch/rd_rl/slurm-%j.log
#SBATCH --chdir=/home/allenw/repos/PopDownGym/pop_down_gym/

date;hostname;id;pwd

srun ./start_agent.sh
EOT