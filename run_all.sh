#! /bin/bash

# Usage:
# ./run_all.sh DATANAME TRANSFORM MODEL
# DATANAME: all TU datasets
# TRANSFORM: hash, diff_one, counts
# MODEL: GIN (for hash), DiffGIN (for diff_one), DeepSet (for hash), DeepCount (for counts)
# example usage: ./run_all.sh MUTAG hash GIN
# ./run_all.sh MUTAG diff_one DiffGIN
# ./run_all.sh MUTAG hash DeepSet
# ./run_all.sh MUTAG counts DeepCount

data=${1}
trans=${2}
model=${3}
for i in $(seq 1 5)  # to run with different seeds
do
  python train.py --transform ${trans} --dataset ${data} --model ${model}
done
