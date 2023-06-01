#!/bin/bash

export ROOT="DUB"
export task="translate"
export src_lang="en_units"
export tgt_lang="de"
export bt_strategy=""    #["", "beam5", "topk10", "topk300"]

#parsing arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in 
        -t|--task) export task="$2"; shift;;
        -src|--src_lang) export src_lang="$2"; shift;;   
        -tgt|--tgt_lang) export tgt_lang="$2"; shift;;   
        -bts|--bt_strategy) export bt_strategy="$2"; shift;;
        *) echo "Unknown parameter passed: $1";;
    esac
    shift
done


for l in ${src_lang} ${tgt_lang};do
    if [[ ! $l =~ "_" ]]; then
        export LANG="${l%%-*}"
    fi
done




bash $ROOT/src/run_${task}.sh

