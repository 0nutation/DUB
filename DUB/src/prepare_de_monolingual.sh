#!/bin/bash

SUBSAMPLE_SIZE=10000000
LANG=de

ROOT="DUB/"
OUTDIR=${ROOT}/data-bin/wmt18_${LANG}_mono
RAW=${OUTDIR}/raw
tmp=$OUTDIR/tmp
SCRIPTS_ROOT=${ROOT}/scripts
mkdir -p $OUTDIR $tmp 
mkdir -p $RAW


URLS=(
    "http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2007.de.shuffled.gz"
    "http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2008.de.shuffled.gz"
    "http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2009.de.shuffled.gz"
    "http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2010.de.shuffled.gz"
    "http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2011.de.shuffled.gz"
    "http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2012.de.shuffled.gz"
    "http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2013.de.shuffled.gz"
    "http://www.statmt.org/wmt15/training-monolingual-news-crawl-v2/news.2014.de.shuffled.v2.gz"
    "http://data.statmt.org/wmt16/translation-task/news.2015.de.shuffled.gz"
    "http://data.statmt.org/wmt17/translation-task/news.2016.de.shuffled.gz"
    "http://data.statmt.org/wmt18/translation-task/news.2017.de.shuffled.deduped.gz"
)
FILES=(
    "news.2007.de.shuffled.gz"
    "news.2008.de.shuffled.gz"
    "news.2009.de.shuffled.gz"
    "news.2010.de.shuffled.gz"
    "news.2011.de.shuffled.gz"
    "news.2012.de.shuffled.gz"
    "news.2013.de.shuffled.gz"
    "news.2014.de.shuffled.v2.gz"
    "news.2015.de.shuffled.gz"
    "news.2016.de.shuffled.gz"
    "news.2017.de.shuffled.deduped.gz"
)


cd $RAW
for ((i=0;i<${#URLS[@]};++i)); do
    file=${FILES[i]}
    if [ -f $file ]; then
        echo "$file already exists, skipping download"
    else
        url=${URLS[i]}
        wget "$url"
    fi
done
cd -


if [ -f $tmp/monolingual.${SUBSAMPLE_SIZE}.${LANG} ]; then
    echo "found monolingual sample, skipping shuffle/sample/tokenize"
else
    gzip -c -d -k $(for FILE in "${FILES[@]}"; do echo $RAW/$FILE; done) \
    | shuf -n $SUBSAMPLE_SIZE \
    > $tmp/monolingual.${SUBSAMPLE_SIZE}.${LANG}
fi


if [ -f $tmp/monolingual.dedup.${SUBSAMPLE_SIZE}.${LANG} ]; then
    echo "found deduplicated monolingual sample, skipping deduplication step"
else
    python3 ${SCRIPTS_ROOT}/deduplicate_lines.py $tmp/monolingual.${SUBSAMPLE_SIZE}.${LANG} \
    > ${OUTDIR}/monolingual.dedup.${SUBSAMPLE_SIZE}.${LANG}
fi
