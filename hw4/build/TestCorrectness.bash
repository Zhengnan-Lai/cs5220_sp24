#!/bin/bash

module load python
srun -N 2 --ntasks-per-node=8 ./mpi -s 777 -o sout777
~/HW2_correctness/correctness-check.py sout777 ../sout777
srun -N 2 --ntasks-per-node=8 ./mpi -s 666 -o sout666
~/HW2_correctness/correctness-check.py sout666 ../sout666
srun -N 2 --ntasks-per-node=8 ./mpi -s 888 -o sout888
~/HW2_correctness/correctness-check.py sout888 ../sout888
srun -N 2 --ntasks-per-node=8 ./mpi -s 7777 -o sout7777 -n 10000
~/HW2_correctness/correctness-check.py sout7777 ../sout7777