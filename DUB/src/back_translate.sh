#!/bin/bash

#-----------------------------------------------Parse Args----------------------------------------------
real_lang="de"
sys_lang="en_units"
bt_strategy="topk10"
seed="2023"
shard_size="100000"

while [[ "$#" -gt 0 ]]; do
    case $1 in 
        --real_lang) export real_lang="$2"; shift;;  #["de","fr","es"]
        --sys_lang) export sys_lang="$2"; shift;;  #[units]
        --ckpt_name) export ckpt_name="$2"; shift;;   # root of back-translate ckpt & spm model
        --bt_strategy) export bt_strategy="$2"; shift;;  #["beam5", "topk10", "topk300"]
        --seed) export seed="$2";shift;;
        --shard_size) export shard_size="$2";shift;;
        *) echo "Unknown parameter passed: $1";;
    esac
    shift
done

for l in ${real_lang} ${sys_lang};do
    if [[ ! $l =~ "_" ]]; then
        export LANG="${l%%-*}"
    fi
done

#-----------------------------------------------CONFIG----------------------------------------------
export ROOT="DUB"
export RAWDATA_ROOT="${ROOT}/data-bin/RawDATA/backtranslate/${real_lang}-${sys_lang}"
export MONOBINARYDATA_ROOT="${ROOT}/data-bin/MONOBinaryDATA/${lang_pair}"
export PARADATA_ROOT="${ROOT}/data-bin/BinaryDATA/translate/${real_lang}-${sys_lang}"
export SCRIPTS_ROOT=${ROOT}/scripts

export spm_model="${RAWDATA_ROOT}/spm.model"
export OUTPUT_DIR=${ROOT}/output/backtranslate/${bt_strategy}

mkdir -p $OUTPUT_DIR

case ${bt_strategy} in 
    "beam5")
        export bt_args="--beam 5"
        ;;
    "topk10")
        export bt_args="--beam 1 --sampling --sampling-topk 10"
        ;;
    "topk300")
        export bt_args="--beam 1 --sampling --sampling-topk 300"
        ;;
esac



ckpt_path="${ROOT}/output/translate/${real_lang}-${sys_lang}/checkpoints/${ckpt_name}"

cp ${ROOT}/data-bin/RawDATA/translate/en_units-${LANG}/spm* ${RAWDATA_ROOT}





#-----------------------------------------------generate----------------------------------------------
bash ${SCRIPTS_ROOT}/prepare_de_monolingual.sh
mv ${MONODATA_ROOT}/monolingual.dedup.40000000.de ${RAWDATA_ROOT}/train.${real_lang} 
# cp ${ROOT}/data-bin/RawDATA/bimodalBART/en_units-de/train.de ${RAWDATA_ROOT}/train.${real_lang} 


# apply spm
python3 ${ROOT}/scripts/apply_spm.py --model ${spm_model} --input-file ${RAWDATA_ROOT}/train.${real_lang} --output-file ${RAWDATA_ROOT}/train.spm.${real_lang} --add_lang_tag ${real_lang}

mkdir -p ${RAWDATA_ROOT}/shards/


#split into shards
if [ -f ${RAWDATA_ROOT}/shards/train.shard00.${real_lang} ]; then
    echo "found sharded data, skipping sharding step"
else
    split -${shard_size} --numeric-suffixes --additional-suffix .${real_lang} \
    ${RAWDATA_ROOT}/train.spm.${real_lang} \
    ${RAWDATA_ROOT}/shards/train.spm.shard
fi

export shards_num=$(ls ${RAWDATA_ROOT}/shards/ | wc -l)
echo "dataset is split into ${shards_num} shards"


#binary
for shard in $(seq -f "%2g" 0 ${shards_num});do
    fairseq-preprocess \
        --only-source \
        --source-lang ${real_lang} --target-lang ${sys_lang} \
        --joined-dictionary \
        --testpref ${RAWDATA_ROOT}/shards/train.spm.shard0${shard} \
        --destdir ${MONOBINARYDATA_ROOT}/shard${shard} \
        --srcdict "${PARADATA_ROOT}/dict.${real_lang}.txt" \
        --workers 100
    cp ${PARADATA_ROOT}/dict.${sys_lang}.txt ${MONOBINARYDATA_ROOT}/shard${shard}/
done


#generate
for shard in $(seq -f "%2g" 0 ${shards_num});do
    CUDA_VISIBLE_DEVICES=${shard} fairseq-generate ${MONOBINARYDATA_ROOT}/shard${shard} \
        --path ${ckpt_path} \
        --skip-invalid-size-inputs-valid-test \
        --max-tokens 20000 \
        --max-len-b 1000 \
        --max-source-positions 2048 --max-target-positions 2048 \
        --remove-prefix-lang-tag "<lang:${sys_lang}>" \
        ${bt_args} \
        --eval-bleu --eval-bleu-bpe sentencepiece --eval-bleu-bpe-path ${spm_model} \
        --seed ${seed} \
        > $OUTPUT_DIR/sampling.shard${shard}.out 
done


#post-process
python3 ${ROOT}/scripts/extract_bt_data.py \
    --minlen 1 --maxlen 500 \
    --output $OUTPUT_DIR/bt_data --srclang ${sys_lang} --tgtlang ${real_lang} \
    $OUTPUT_DIR/sampling.shard*.out


sed -i 's/^/<BT>&/g' $OUTPUT_DIR/bt_data.${sys_lang}


mkdir -p ${ROOT}/data-bin/BTDATA/${bt_strategy}
cp $OUTPUT_DIR/bt_data.${sys_lang} ${ROOT}/data-bin/BTDATA/${bt_strategy}/train.bt.${sys_lang} 
cp $OUTPUT_DIR/bt_data.${real_lang} ${ROOT}/data-bin/BTDATA/${bt_strategy}/train.bt.${real_lang} 



#generate
# bash DUB/src/back_translate.sh --real_lang de --sys_lang en_units --ckpt_name checkpoint_best.pt --bt_strategy topk10