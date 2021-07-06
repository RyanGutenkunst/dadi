#!/bin/bash -l
#
# this is an example slurm script for sbatch
# you will need to make a couple of changes
# 1. replace ACCOUNT with your HPC account name (two places)
# 2. edit parameters to dd-workqueue.py as needed
# 3. changed SBATCH --nodes and -ntasks values to be --jobs value + 1
# 4. if you use the -P parameter for work_queue_factory, create your password file
#
#SBATCH --account=ACCOUNT
#SBATCH --partition=standard
#SBATCH --job-name="dadi_worker"
#SBATCH --nodes=3
#SBATCH --ntasks=3
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00

export CCTOOLS_HOME=${HOME}/.conda/envs/py37
export PATH=${CCTOOLS_HOME}/bin:$PATH

~/dadi/dadi-distributed/dd-workqueue.py --model split_mig --misid --p0 "1 1 .5 1 .5"  --ubounds "10 10 1 10 1"  --lbounds "10e-3 0 10e-3 10e-3 10e-5" --jobs 2 run ~/dadi-cli/examples/results/fs/1KG.YRI.CEU.proj.fs &

~/.conda/envs/py37/bin/work_queue_factory -T slurm -M dadi-distributed -P pwfile -B "--account=ACCOUNT --partition=standard --job-name=dadi_worker"  --workers-per-cycle 0 --cores=2
