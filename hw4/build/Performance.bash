#!/bin/bash

srun -N 1 --ntasks-per-node=1 ./mpi -n 1000 -s 10
srun -N 1 --ntasks-per-node=1 ./mpi -n 10000 -s 10
srun -N 1 --ntasks-per-node=1 ./mpi -n 100000 -s 10
srun -N 1 --ntasks-per-node=64 ./mpi -n 2000000 -s 10
srun -N 2 --ntasks-per-node=64 ./mpi -n 2000000 -s 10
srun -N 2 --ntasks-per-node=128 ./mpi -n 2000000 -s 10