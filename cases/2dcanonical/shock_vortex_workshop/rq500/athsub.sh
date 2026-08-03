#PBS -S /bin/csh
#PBS -N FR_PPR_SVI
#PBS -q normal
#PBS -l walltime=6:00:00
#PBS -l select=1:ncpus=256:mpiprocs=256:model=tur_ath
#PBS -j oe
#PBS -W group_list=a1462
#PBS -o jobout.log

###  qsub -I -N GFR_SHORTJET -q devel -l walltime=1:00:00 -l select=20:ncpus=128:mpiprocs=128:model=rom_ait

# By default, PBS executes your job from your home directory. However, you can
# use the environment variable PBS_O_WORKDIR to change to the directory where
# you submitted your job.
cd "${PBS_O_WORKDIR}"

# This line is only needed if this doesnt exist in your .cshrc file
module use /u/sspiegel/gfr/modulefiles/cray

# Load the necessary modules
# NOTE: You will need to change the executable and its path to where
# you configured the system_variables.mk file to copy your executables.
module purge
module load gfr


# Optional: output the complete job information to the output file
qstat -fJt "${PBS_JOBID}"     >>&! out.log

./run_case.sh -headless -clean >> out.log

# -end of script-
