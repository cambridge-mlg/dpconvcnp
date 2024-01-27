#!/bin/bash

for epochs in 1 3 10 30 100 300 1000
do
    python experiments/time_dpsgp.py --config experiments.config/dpsgp-time.yml params.epochs=$epochs
done