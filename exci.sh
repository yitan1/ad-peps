#!/bin/bash

for((i=1;i<=31;i++));  
do   
python -m adpeps exci ising_D2_exci.yaml -p $i 
done  