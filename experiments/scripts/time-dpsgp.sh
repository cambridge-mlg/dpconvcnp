#!/bin/bash

for epochs in 1 3 10 30 100 300 1000
do
    python experiments/time_dpsgp.py --config experiments/config/dpsgp-time.yml limits.min_epochs=$epochs limits.max_epochs=$epochs misc.name=epochs-$epochs
done