LANG=$1
CKPT_PATH=$2

ROOT="DUB"
SRC_LANG="en_units"
TGT_LANG="${LANG}"
RAWDATA_ROOT="${ROOT}/data-bin/RawDATA/translate/en_units-${LANG}"
BINARYDATA_ROOT="${ROOT}/data-bin/BinaryDATA/translate/${SRC_LANG}-${TGT_LANG}"

OUTPUT_ROOT="${ROOT}/output/evaluate/"


mkdir -p ${OUTPUT_ROOT}

fairseq-generate ${BINARYDATA_ROOT} \
--beam 10 --path ${CKPT_PATH} --lenpen 2 --diversity-rate -0.1 \
--source-lang ${SRC_LANG} --target-lang ${TGT_LANG} \
--max-source-positions 4096 --max-target-positions 4096 \
--remove-prefix-lang-tag "<lang:${TGT_LANG}>" \
--eval-bleu --eval-bleu-bpe sentencepiece --eval-bleu-bpe-path $RAWDATA_ROOT/spm.model > ${OUTPUT_ROOT}/test.log


grep ^D ${OUTPUT_ROOT}/test.log | cut -f3 | sacremoses detokenize > ${OUTPUT_ROOT}/test.sys
grep ^T ${OUTPUT_ROOT}/test.log | cut -f2 | sacremoses detokenize > ${OUTPUT_ROOT}/test.ref


python3 fairseq_cli/score.py --sys ${OUTPUT_ROOT}/test.sys --ref ${OUTPUT_ROOT}/test.ref --sacrebleu > ${OUTPUT_ROOT}/score.log
cat ${OUTPUT_ROOT}/score.log


# bash zhangdong/scripts/sacrebleu.sh data-bin/RawDATA/covost2/${lang}/data data-bin/BinaryDATA/${lang}/units-en/ checkpoints/checkpoints_${lang}-units_en_parallel/checkpoint.best_avg.7109-3.pt 8000 units en discreteX2E