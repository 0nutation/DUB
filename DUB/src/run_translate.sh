# #!/bin/bash
#task : translate


echo "Task is ${src_lang} to ${tgt_lang}"
echo "bt_strategy is ${bt_strategy}"


##################################### dir ####################################
export RAWDATA_ROOT="${ROOT}/data-bin/RawDATA/${task}/en_units-${LANG}"
export BINARYDATA_ROOT="${ROOT}/data-bin/BinaryDATA/${task}/${src_lang}-${tgt_lang}"
export OUTPUT_ROOT="${ROOT}/output/${task}/${src_lang}-${tgt_lang}"
export CHECKPOINT_DIR="$OUTPUT_ROOT/checkpoints/"
export LOG_ROOT="$OUTPUT_ROOT/logs/"
export SCRIPTS_ROOT="${ROOT}/scripts"

export BTDATA_ROOT="${ROOT}/data-bin/BTDATA/${bt_strategy}"

export vocab_size=8000
export units_size=500



export MBART_DIR="${ROOT}/utils/bimodal_bart"
# export BIMODAL_BART_INIT="False"


if [[ ${BIMODAL_BART_INIT} = "True" ]];then
    if [ ! -f ${MBART_DIR}/bimodal_bart.pt ];then
        mkdir -p ${MBART_DIR}
        cp ${ROOT}/output/bimodalBART/en_units-${LANG}/checkpoint_best.pt ${MBART_DIR}/bimodal_bart.pt
    fi
    export lpfARGS="--load-pretrained-encoder-from ${MBART_DIR}/bimodal_bart.pt --load-pretrained-decoder-from ${MBART_DIR}/bimodal_bart.pt"   
fi


if [[ ${tgt_lang} =~ "units" ]];then
    best_metric="accuracy"
else
    best_metric="bleu"
fi




# ------------------------------------------PREPROCESS--------------------------------------------

for l in ${src_lang} ${tgt_lang};do
    if [[ ! $l =~ "_" ]]; then
        cat ${RAWDATA_ROOT}/train.${l} >> ${RAWDATA_ROOT}/train.mix
    fi
done


#train spm
python3 ${SCRIPTS_ROOT}/train_spm.py \
    --input-path ${RAWDATA_ROOT}/train.mix \
    --model-prefix ${RAWDATA_ROOT}/spm \
    --vocab-size ${vocab_size} \
    --units-size ${units_size} \
    --lang "${src_lang} ${tgt_lang}" \
    --units-prefix "#0"

    


# external back translation pseudo data
if [[ ${bt_strategy} != "" ]];then
    cat ${BTDATA_ROOT}/train.bt.${src_lang} >> ${RAWDATA_ROOT}/train.${src_lang}
    cat ${BTDATA_ROOT}/train.bt.${tgt_lang} >> ${RAWDATA_ROOT}/train.${tgt_lang}
fi


#tokenize and binary data
for split in train dev tst-COMMON;do
    for lang in ${src_lang} ${tgt_lang};do
        python3 ${SCRIPTS_ROOT}/apply_spm.py \
            --model ${RAWDATA_ROOT}/spm.model \
            --input-file ${RAWDATA_ROOT}/${split}.${lang} --output-file ${RAWDATA_ROOT}/${split}.spm.${lang} \
            --add_lang_tag ${lang}
        done
    done


fairseq-preprocess \
    --source-lang ${src_lang} --target-lang ${tgt_lang} \
    --trainpref ${RAWDATA_ROOT}/train.spm --validpref ${RAWDATA_ROOT}/dev.spm --testpref ${RAWDATA_ROOT}/tst-COMMON.spm \
    --destdir ${BINARYDATA_ROOT} \
    --srcdict ${RAWDATA_ROOT}/spm.txt --tgtdict ${RAWDATA_ROOT}/spm.txt \
    --workers 100


# -----------------------------------------------TRAIN-------------------------------------------

mkdir -p ${CHECKPOINT_DIR}
mkdir -p $LOG_ROOT/


# fairseq-train \
fairseq-train --fp16 \
    ${BINARYDATA_ROOT}/ \
    --source-lang ${src_lang} --target-lang ${tgt_lang} \
    --arch transformer_wmt_en_de_kmemb \
    --dropout 0.3 --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 10.0 \
    --lr-scheduler inverse_sqrt  \
    --max-update 3000000 --truncate-source \
    --save-dir $CHECKPOINT_DIR \
    --eval-bleu --eval-bleu-args '{"beam": 5, "prefix_size": 1}' \
    --eval-bleu-detok moses --eval-bleu-remove-bpe \
    --eval-bleu-bpe sentencepiece --eval-bleu-bpe-path $RAWDATA_ROOT/spm.model \
    --best-checkpoint-metric ${best_metric} --maximize-best-checkpoint-metric --eval-bleu-print-samples \
    --tensorboard-logdir $LOG_ROOT/ --remove-prefix-lang-tag "<lang:${tgt_lang}>" --report-accuracy \
    --log-interval 100 \
    --save-interval 1 \
    --keep-last-epochs 10 --keep-interval-updates 100 --keep-best-checkpoints 10 \
    --fp16-scale-tolerance=0.25 \
    --lr 0.001 --max-tokens 4000 --warmup-updates 4000 --max-source-positions 4096 --max-target-positions 4096 --update-freq 8 --save-interval-updates 600 \
    --encoder-layers 12 --encoder-normalize-before \
    ${lpfARGS} \

