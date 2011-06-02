# configuration file. Should be sourced from the main simulation script

# define all files
defaults=defaults # input file
options=options # input file
sample=sample.txt # input file
params=params.txt # input file
index=index.txt
clusterconf=cluster_conf.py # input file
clusterlog=cluster.log
data=data.npz

# user parameters, default values
prefix=out # output file will have this prefix
reps=1 # will perform these many runs for each line in the sample
usessh=0 # if 1, ipcluster will be called with `ssh` mode, else `local`
overwrite=0 # if 1, will overwrite the output file, if it exists

# define getopt's option strings
SHORT_OPTS=r:p:c:sho
LONG_OPTS=reps:,prefix:,cpus:,ssh,help,overwrite

