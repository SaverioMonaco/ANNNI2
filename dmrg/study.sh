#!/usr/bin/env bash

pL=(4 8 12 16 32)
pX=(2 4 8 16 32 64 )
side=40

for L in "${pL[@]}"
    do
        for X in "${pX[@]}"
            do 
                echo "L = $L, X = $X"
                python3 getstates.py --side $side --L $L --chi $X --hide
            done
    done
