#!/bin/bash

module load python
./gpu -s 777 -o sout777
~/HW2_correctness/correctness-check.py sout777 ../sout777
./gpu -s 666 -o sout666
~/HW2_correctness/correctness-check.py sout666 ../sout666
./gpu -s 888 -o sout888
~/HW2_correctness/correctness-check.py sout888 ../sout888
# ../serial -s 7777 -n 10000 -o ../sout7777
./gpu -s 7777 -n 10000 -o sout7777
~/HW2_correctness/correctness-check.py sout7777 ../sout7777