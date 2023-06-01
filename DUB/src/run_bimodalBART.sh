# #!/bin/bash
#task : pretrain bimodal BART


echo "Task is to pretrain ${src_lang}-${tgt_lang} bimodal BART"

#-----------------------------------------------CONFIG----------------------------------------------
export RAWDATA_ROOT="${ROOT}/data-bin/RawDATA/${task}/${src_lang}-${tgt_lang}"
export BINARYDATA_ROOT="${ROOT}/data-bin/BinaryDATA/${task}/${src_lang}-${tgt_lang}"
export OUTPUT_ROOT="${ROOT}/output/${task}/${src_lang}-${tgt_lang}"
export CHECKPOINT_DIR="$OUTPUT_ROOT/checkpoints/"
export LOG_ROOT="$OUTPUT_ROOT/logs/"
export SCRIPTS_ROOT="${ROOT}/scripts"
export MONODATA_ROOT="${ROOT}/data-bin/wmt18_de_mono"


export vocab_size=8000
export units_size=500


mkdir -p ${RAWDATA_ROOT}
mkdir -p ${CHECKPOINT_DIR}
mkdir -p ${LOG_ROOT}

#prepare unlabeled text and unit data
bash ${SCRIPTS_ROOT}/prepare_de_monolingual.sh
cp ${MONODATA_ROOT}/monolingual.dedup.10000000.de ${RAWDATA_ROOT}/train.de

cp ${ROOT}/data-bin/RawDATA/translate/${src_lang}-${tgt_lang}/dev.de ${RAWDATA_ROOT}/dev.de
cp ${ROOT}/data-bin/RawDATA/translate/${src_lang}-${tgt_lang}/dev.en_units ${RAWDATA_ROOT}/dev.en_units

#TODO: write on readme
# generate gigaspeech units to ${RAWDATA_ROOT}/train.units


cp ${ROOT}/data-bin/RawDATA/translate/${src_lang}-${tgt_lang}/spm* ${RAWDATA_ROOT}/


for la in ${src_lang} ${tgt_lang};do
    for split in train dev;do
        echo "applying spm for train/dev.${la}"
        python3 ${SCRIPTS_ROOT}/apply_spm.py \
            --model ${RAWDATA_ROOT}/spm.model \
            --input-file ${RAWDATA_ROOT}/${split}.${la} \
            --output-file ${RAWDATA_ROOT}/${split}.spm.${la} \
            --add_lang_tag ${la} 
        done

    echo "binary train/dev.spm.${la}"
    mkdir -p ${BINARYDATA_ROOT}
    fairseq-preprocess \
        --only-source \
        --trainpref ${RAWDATA_ROOT}/train.spm.${la} \
        --validpref ${RAWDATA_ROOT}/dev.spm.${la} \
        --destdir ${BINARYDATA_ROOT}/${la}/ \
        --srcdict ${RAWDATA_ROOT}/spm.txt \
        --workers 100 
done


cp ${BINARYDATA_ROOT}/${tgt_lang}/dict.txt ${BINARYDATA_ROOT}/dict.txt


# -----------------------------------------------TRAIN-------------------------------------------

fairseq-train --fp16 \
    ${BINARYDATA_ROOT}/ \
    --task multilingual_denoising \
    --langs "${tgt_lang}" \
    --arch mbart_kmemb \
    --mask 0.3 --mask-length span-poisson --replace-length 1 --mask-random 0.1 \
    --rotate 0.0 --permute-sentences 1 --insert 0.0 --poisson-lambda 3.5 \
    --tokens-per-sample 1020 \
    --total-num-update 1000000 --max-update 1000000 \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr-scheduler polynomial_decay \
    --criterion cross_entropy \
    --dropout 0.1 --weight-decay 0.01 --attention-dropout 0.1 --clip-norm 0.1 \
    --share-all-embeddings --skip-invalid-size-inputs-valid-test \
    --log-format json \
    --save-interval 1  --save-interval-updates 1000 --keep-interval-updates 10 \
    --seed 2 \
    --num-workers 4 \
    --save-dir $CHECKPOINT_DIR \
    --tensorboard-logdir $LOG_ROOT/ \
    --fp16-scale-tolerance=0.25 \
    --lr 0.0004 --max-tokens 4096 --update-freq 4 --max-source-positions 2048 --max-target-positions 2048 --warmup-updates 10000 --log-interval 1000 



