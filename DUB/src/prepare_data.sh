LANG=$1
ROOT="DUB"

MUSTC_ROOT="${ROOT}/data-bin/MuSTC"
HUBERT_DIR="${ROOT}/utils/hubert"
OUTPUT_DIR="${ROOT}/data-bin/RawDATA/translate/en_units-${LANG}"

mkdir -p $MUSTC_ROOT
mkdir -p ${OUTPUT_DIR}
mkdir -p ${HUBERT_DIR}

#download hubert
if [ ! -f ${HUBERT_DIR}/hubert_model.pt ];then
    wget https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt -O ${HUBERT_DIR}/hubert_model.pt
    wget https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960_L9_km500.bin -O ${HUBERT_DIR}/km.model
fi


#extract units and tsv files
echo "extracting training data"
python3 ${ROOT}/scripts/prepare_data.py \
    --data-root ${MUSTC_ROOT} \
    --language ${LANG} \
    --hubert-dir ${HUBERT_DIR} \
    --output-dir ${OUTPUT_DIR}
