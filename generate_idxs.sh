#!/bin/bash
#DIR=robin_mesh_3D 
for test in 4
do
    TEST="test${test}"
    for refine in {0..3}
    do
        DIR="${TEST}/refine${refine}"
        for i in {1..4}
        do
            ifile="build/${DIR}/BDDC_I$i.txt"
            tail -n 1 ${ifile} | grep -Eo "[0-9]+" > build/${DIR}/local_idxs_$i.txt
            bfile="build/${DIR}/BDDC_Gamma$i.txt"
            tail -n 1 ${bfile} | grep -Eo "[0-9]+" >> build/${DIR}/local_idxs_$i.txt
        done
    done
done
