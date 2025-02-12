#!/bin/sh
#
#SBATCH --job-name="muscle"
#SBATCH --partition=compute
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=3G
#SBATCH --account=research-as-bt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=B.Magyar@student.tudelft.nl

# Input/output paths
INPUT=$HOME/nitroplast-new/lost_proteins.fasta
OUTPUT=$HOME/nitroplast-new/lost_sequences_aligned.fa

# Activate environment if needed
eval "$(micromamba shell hook --shell bash)"
micromamba activate muscle

echo "Running MUSCLE alignment..."
muscle -align $INPUT -output $OUTPUT -threads 10

echo "All done!"

micromamba deactivate