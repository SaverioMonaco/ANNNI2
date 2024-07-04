#!/usr/bin/env bash

pL=(4 8 12 16   )
pX=(64 )
side=20

for L in "${pL[@]}"
    do
        for X in "${pX[@]}"
            do 
                echo "L = $L, X = $X"
                python3 getstates.py --side $side --L $L --chi $X --hide
            done
    done
