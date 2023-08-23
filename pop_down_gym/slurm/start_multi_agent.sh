#!/bin/bash
sbatch <<EOT
#!/bin/bash

#SBATCH --nodes=4               # number of nodes
#SBATCH --ntasks-per-node=1     # MPI processes per node
#SBATCH --partition=sched_mit_nse
#SBATCH --time=0-04:00:0
#SBATCH --output=/home/allenw/Scratch/rd_rl/slurm-%j.log
#SBATCH --chdir=/home/allenw/repos/PopDownGym/pop_down_gym/slurm

date;hostname;id;pwd
srun ./start_agent.sh
EOT