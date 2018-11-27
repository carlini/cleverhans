# Usage:
# Same as usage of pgd.py, but don't specify eps or nb_iter.
# This script calls through to pgd.py with several values of eps.

# Values of eps to use
declare -a eps_values=(".01" ".1" ".2" ".4" ".5")

# Quit on first error
set -e

# Find the directory this script is in
dir=`dirname $0`
# pgd.py is in the same directory
pgd=${dir}/pgd.py

# Run once per epsion value
for eps in "${eps_values[@]}"
do
  # We run the attack for only 100 iterations because this step of the checklist is not intended to
  # be maximally strong, just to make sure that the attack responds reasonably to eps.
  python $pgd $@ --eps ${eps} --nb_iter 100
done

