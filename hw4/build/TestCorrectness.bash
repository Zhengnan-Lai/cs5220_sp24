#!/bin/bash

module load python
# srun -N 1 --ntasks-per-node=1 ./mpi -s 777
# srun -N 1 --ntasks-per-node=1 ./mpi -n 100000
srun -N 1 --ntasks-per-node=1 ./mpi -s 777 -o sout777
~/HW2_correctness/correctness-check.py sout777 ../sout777
srun -N 1 --ntasks-per-node=1 ./mpi -s 666 -o sout666
~/HW2_correctness/correctness-check.py sout666 ../sout666
srun -N 1 --ntasks-per-node=1 ./mpi -s 888 -o sout888
~/HW2_correctness/correctness-check.py sout888 ../sout888
srun -N 1 --ntasks-per-node=1 ./mpi -s 7777 -o sout7777 -n 10000
~/HW2_correctness/correctness-check.py sout7777 ../sout7777
srun -N 1 --ntasks-per-node=1 ./mpi -s 1 -o sout1e5 -n 100000
~/HW2_correctness/correctness-check.py sout1e5 ../sout1e5