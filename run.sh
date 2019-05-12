#!/bin/bash

set -e

##### Please edit this section to fit your experiment setting

DATA_PATH=          # corpus path (source corpus named as `pro` and target corpus as `con`)
EMB_DIM=300         # dimension of the professional (pro) and consumer (con) word embeddings
SUBWORD=false       # whether to use subword information when training monolingual word embeddings
USE_NEWSCRAWL=false # whether to use third-party large-scale corpora for training con LM
N_THREADS=48        # number of threads in data preprocessing
IDENTICAL_CHAR=true # whether to use identical chars as anchors when training cross-lingual embeddings
RESULT_DIR=test     # where the experimental results will be stored
SENT_BT=100000      # monolingual data used in back-translation


##### No need to edit the following sections for default setting

WD=$(pwd)
TRAIN_PATH=$WD/res/$RESULT_DIR # main path of the experiment

# create paths
mkdir -p $DATA_PATH
mkdir -p $TRAIN_PATH

# --- Download and install tools
# fastText
FASTTEXT_EXE=./fastText-0.2.0/fasttext

if [ ! -f "$FASTTEXT_EXE" ]; then
  echo "Installing fastText from source..."
  wget https://github.com/facebookresearch/fastText/archive/v0.2.0.zip
  unzip v0.2.0.zip
  cd fastText-0.2.0
  make
  rm ../v0.2.0.zip
  cd ..
fi
echo "fastText found in: $FASTTEXT_EXE"

# MUSE
MUSE_PATH=$WD/MUSE

if [ ! -d "$MUSE_PATH" ]; then
  echo "Cloning MUSE from GitHub repository..."
  git clone https://github.com/facebookresearch/MUSE.git
  cd $MUSE_PATH/data/
fi
echo "MUSE found in: $MUSE_PATH"

# UMT
UMT_PATH=$WD/UnsupervisedMT/PBSMT

if [ ! -d "$UMT_PATH" ]; then
  echo "Cloning UnsuperviedMT from GitHub repository..."
  git clone https://github.com/facebookresearch/UnsupervisedMT.git
fi
echo "UnsuperviedMT found in: $UMT_PATH"

# MOSES
MOSES_PATH=$WD/ubuntu-17.04/moses

if [ ! -d "$MOSES_PATH" ]; then
  echo "Getting MOSES..."
  wget http://www.statmt.org/moses/RELEASE-4.0/binaries/ubuntu-17.04.tgz
  tar -zxvf ubuntu-17.04.tgz
fi
echo "MOSES found in: $MOSES_PATH"

TOKENIZER=$MOSES_PATH/scripts/tokenizer/tokenizer.perl
NORM_PUNC=$MOSES_PATH/scripts/tokenizer/normalize-punctuation.perl
INPUT_FROM_SGM=$MOSES_PATH/scripts/ems/support/input-from-sgm.perl
REM_NON_PRINT_CHAR=$MOSES_PATH/scripts/tokenizer/remove-non-printing-char.perl
TRAIN_TRUECASER=$MOSES_PATH/scripts/recaser/train-truecaser.perl
TRUECASER=$MOSES_PATH/scripts/recaser/truecase.perl
DETRUECASER=$MOSES_PATH/scripts/recaser/detruecase.perl
TRAIN_LM=$MOSES_PATH/bin/lmplz
TRAIN_MODEL=$MOSES_PATH/scripts/training/train-model.perl
MULTIBLEU=$MOSES_PATH/scripts/generic/multi-bleu.perl
MOSES_BIN=$MOSES_PATH/bin/moses

# files full paths
SRC_RAW=$DATA_PATH/pro
SRC_TOK=$TRAIN_PATH/pro.tok
SRC_TRUE=$TRAIN_PATH/pro.true
SRC_TRUECASER=$TRAIN_PATH/pro.truecaser

if [ "$USE_NEWSCRAWL" = true ]; then
  N_MONO=10000000
  TGT_RAW=$TRAIN_PATH/all
  TGT_TOK=$TRAIN_PATH/all.tok
  TGT_TRUE=$TRAIN_PATH/all.true
  TGT_TRUECASER=$TRAIN_PATH/all.truecaser
else
  TGT_RAW=$DATA_PATH/con
  TGT_TOK=$TRAIN_PATH/con.tok
  TGT_TRUE=$TRAIN_PATH/con.true
  TGT_TRUECASER=$TRAIN_PATH/con.truecaser
fi

SRC_TEST=$DATA_PATH/pro
SRC_LM_ARPA=$TRAIN_PATH/pro.lm.arpa
TGT_LM_ARPA=$TRAIN_PATH/con.lm.arpa
SRC_LM_BLM=$TRAIN_PATH/pro.lm.blm
TGT_LM_BLM=$TRAIN_PATH/con.lm.blm

# Check Moses files
if ! [[ -f "$TOKENIZER" && -f "$NORM_PUNC" && -f "$INPUT_FROM_SGM" && -f "$REM_NON_PRINT_CHAR" && -f "$TRAIN_TRUECASER" && -f "$TRUECASER" && -f "$DETRUECASER" && -f "$TRAIN_MODEL" ]]; then
  echo "Some Moses files were not found."
  echo "Please update the MOSES variable to the path where you installed Moses."
  exit
fi
if ! [[ -f "$MOSES_BIN" ]]; then
  echo "Couldn't find Moses binary in: $MOSES_BIN"
  echo "Please check your installation."
  exit
fi
if ! [[ -f "$TRAIN_LM" ]]; then
  echo "Couldn't find language model trainer in: $TRAIN_LM"
  echo "Please install KenLM."
  exit
fi


# --- Train monolingual word embeddings

EMB_SRC=$TRAIN_PATH/pro
EMB_TGT=$TRAIN_PATH/con

if [ "$SUBWORD" = true ]; then
 $FASTTEXT_EXE skipgram -input $DATA_PATH/pro -output $EMB_SRC -minn 2 -maxn 5 -dim $EMB_DIM -thread 12
 $FASTTEXT_EXE skipgram -input $DATA_PATH/con -output $EMB_TGT -minn 2 -maxn 5 -dim $EMB_DIM -thread 12
else
 $FASTTEXT_EXE skipgram -input $DATA_PATH/pro -output $EMB_SRC -maxn 0 -dim $EMB_DIM -thread 12
 $FASTTEXT_EXE skipgram -input $DATA_PATH/con -output $EMB_TGT -maxn 0 -dim $EMB_DIM -thread 12
fi

echo "Pretrained pro embeddings found in: $EMB_SRC"
echo "Pretrained con embeddings found in: $EMB_TGT"

EMB_SRC=$EMB_SRC.vec
EMB_TGT=$EMB_TGT.vec


# --- Download monolingual data

cd $DATA_PATH

echo "Downloading English files..."
wget -c http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2007.en.shuffled.gz
wget -c http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2008.en.shuffled.gz
wget -c http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2009.en.shuffled.gz
wget -c http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2010.en.shuffled.gz
#wget -c http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2011.en.shuffled.gz
#wget -c http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2012.en.shuffled.gz
#wget -c http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2013.en.shuffled.gz
#wget -c http://www.statmt.org/wmt15/training-monolingual-news-crawl-v2/news.2014.en.shuffled.v2.gz
#wget -c http://data.statmt.org/wmt16/translation-task/news.2015.en.shuffled.gz
#wget -c http://data.statmt.org/wmt17/translation-task/news.2016.en.shuffled.gz
#wget -c http://data.statmt.org/wmt18/translation-task/news.2017.en.shuffled.deduped.gz

# decompress monolingual data
for FILENAME in news*gz; do
  OUTPUT="${FILENAME::-3}"
  if [ ! -f "$OUTPUT" ]; then
    echo "Decompressing $FILENAME..."
    gunzip -k $FILENAME
  else
    echo "$OUTPUT already decompressed."
  fi
done

# concatenate monolingual data files
if ! [[ -f "$TGT_RAW" ]]; then
  echo "Concatenating monolingual data..."
  cat $(ls news*fr* | grep -v gz) | head -n $N_MONO > $TGT_RAW
fi
echo "$TGT monolingual data concatenated in: $TGT_RAW"

# tokenize data
echo "Tokenize monolingual data..."
if ! [[ -f "$SRC_TOK" ]]; then
  cat $SRC_RAW | $NORM_PUNC -l en | $TOKENIZER -l en -no-escape -threads $N_THREADS > $SRC_TOK
fi
if ! [[ -f "$TGT_TOK" ]]; then
  cat $TGT_RAW | $NORM_PUNC -l en | $TOKENIZER -l en -no-escape -threads $N_THREADS > $TGT_TOK
fi

echo "pro monolingual data tokenized in: $SRC_TOK"
echo "con monolingual data tokenized in: $TGT_TOK"

# learn truecasers
echo "Learning truecasers..."
if ! [[ -f "$SRC_TRUECASER" ]]; then
  $TRAIN_TRUECASER --model $SRC_TRUECASER --corpus $SRC_TOK
fi
if ! [[ -f "$TGT_TRUECASER" ]]; then
  $TRAIN_TRUECASER --model $TGT_TRUECASER --corpus $TGT_TOK
fi

echo "pro truecaser in: $SRC_TRUECASER"
echo "con truecaser in: $TGT_TRUECASER"

# truecase data
echo "Truecsing monolingual data..."
if ! [[ -f "$SRC_TRUE" ]]; then
  $TRUECASER --model $SRC_TRUECASER < $SRC_TOK > $SRC_TRUE
fi
if ! [[ -f "$TGT_TRUE" ]]; then
  $TRUECASER --model $TGT_TRUECASER < $TGT_TOK > $TGT_TRUE
fi

echo "pro monolingual data truecased in: $SRC_TRUE"
echo "con monolingual data truecased in: $TGT_TRUE"

# learn language models
echo "Learning language models..."
if ! [[ -f "$SRC_LM_ARPA" ]]; then
  $TRAIN_LM -o 5 -S 20% < $SRC_TRUE > $SRC_LM_ARPA
fi
if ! [[ -f "$TGT_LM_ARPA" ]]; then
  $TRAIN_LM -o 5 -S 20% --discount_fallback < $TGT_TRUE > $TGT_LM_ARPA
fi

echo "pro language model in: $SRC_LM_ARPA"
echo "con language model in: $TGT_LM_ARPA"

# binarize language models
echo "Binarizing language models..."
if ! [[ -f "$SRC_LM_BLM" ]]; then
  $MOSES_PATH/bin/build_binary $SRC_LM_ARPA $SRC_LM_BLM
fi
if ! [[ -f "$TGT_LM_BLM" ]]; then
  $MOSES_PATH/bin/build_binary -s $TGT_LM_ARPA $TGT_LM_BLM
fi

echo "pro binarized language model in: $SRC_LM_BLM"
echo "con binarized language model in: $TGT_LM_BLM"


# --- Running MUSE to generate cross-lingual embeddings

ALIGNED_EMBEDDINGS_SRC=$MUSE_PATH/$RESULT_DIR/p2c/vectors-pro.pth
ALIGNED_EMBEDDINGS_TGT=$MUSE_PATH/$RESULT_DIR/p2c/vectors-con.pth

if [[ $IDENTICAL_CHAR = true ]]; then
  if ! [[ -f "$ALIGNED_EMBEDDINGS_SRC" && -f "$ALIGNED_EMBEDDINGS_TGT" ]]; then
    echo "Aligning embeddings with MUSE (identical_char)..."
    python $MUSE_PATH/supervised.py --src_lang pro --tgt_lang con \
    --exp_path $MUSE_PATH --exp_name $RESULT_DIR --exp_id p2c \
    --emb_dim $EMB_DIM \
    --src_emb $EMB_SRC \
    --tgt_emb $EMB_TGT \
    --n_refinement 5 --dico_train identical_char --export "pth" \
    --dico_eval $WD/data/cd_pairs.txt
  fi
else
  if ! [[ -f "$ALIGNED_EMBEDDINGS_SRC" && -f "$ALIGNED_EMBEDDINGS_TGT" ]]; then
    echo "Aligning embeddings with MUSE (unsupervised)..."
    python $MUSE_PATH/unsupervised.py --src_lang pro --tgt_lang con \
    --exp_path $MUSE_PATH --exp_name $RESULT_DIR --exp_id p2c \
    --dis_most_frequent 1000 \
    --emb_dim $EMB_DIM \
    --src_emb $EMB_SRC \
    --tgt_emb $EMB_TGT \
    --n_refinement 5 --export "pth" \
    --dico_eval $WD/data/cd_pairs.txt
  fi
fi

echo "pro aligned embeddings: $ALIGNED_EMBEDDINGS_SRC"
echo "con aligned embeddings: $ALIGNED_EMBEDDINGS_TGT"


# --- Generating a phrase-table in an unsupervised way

PHRASE_TABLE_PATH=$TRAIN_PATH/phrase-table.pro-con.gz
if ! [[ -f "$PHRASE_TABLE_PATH" ]]; then
  echo "Generating unsupervised phrase-table"
  python $UMT_PATH/create-phrase-table.py \
  --src_lang pro \
  --tgt_lang con \
  --src_emb $ALIGNED_EMBEDDINGS_SRC \
  --tgt_emb $ALIGNED_EMBEDDINGS_TGT \
  --csls 1 \
  --max_rank 200 \
  --max_vocab 300000 \
  --inverse_score 1 \
  --temperature 45 \
  --phrase_table_path ${PHRASE_TABLE_PATH::-3}
fi
echo "Phrase-table location: $PHRASE_TABLE_PATH"


# --- Train Moses on the generated phrase-table

echo "Generating Moses configuration in: $TRAIN_PATH"

echo "Creating default configuration file..."
$TRAIN_MODEL -root-dir $TRAIN_PATH \
-f en -e en -alignment grow-diag-final-and -reordering msd-bidirectional-fe \
-lm 0:5:$TGT_LM_BLM:8 -external-bin-dir $MOSES_PATH/tools \
-cores $N_THREADS -max-phrase-length=4 -score-options "--NoLex" -first-step=9 -last-step=9
CONFIG_PATH=$TRAIN_PATH/model/moses.ini

echo "Removing lexical reordering features ..."
mv $TRAIN_PATH/model/moses.ini $TRAIN_PATH/model/moses.ini.bkp
cat $TRAIN_PATH/model/moses.ini.bkp | grep -v LexicalReordering > $TRAIN_PATH/model/moses.ini

echo "Linking phrase-table path..."
ln -sf $PHRASE_TABLE_PATH $TRAIN_PATH/model/phrase-table.gz

echo "Translating test sentences..."
$MOSES_BIN -threads $N_THREADS -f $CONFIG_PATH < $TRAIN_PATH/pro.true > $TRAIN_PATH/test.tgt.hyp.true

echo "Detruecasing hypothesis..."
$DETRUECASER < $TRAIN_PATH/test.tgt.hyp.true > $TRAIN_PATH/test.tgt.hyp.tok


# --- Back-translation

echo "Back Translate procedure..."
epoch=1
TRAIN_DIR_ITER_FORWARD=${TRAIN_PATH}-${epoch}-forward
mkdir -p $TRAIN_DIR_ITER_FORWARD
mkdir -p $TRAIN_DIR_ITER_FORWARD/model

# copy the initial model in first epoch
cp ${CONFIG_PATH} $TRAIN_DIR_ITER_FORWARD/model/moses.ini

echo "Linking phrase-table path..."
ln -sf $PHRASE_TABLE_PATH $TRAIN_DIR_ITER_FORWARD/model/phrase-table.gz

for epoch in {1..2}; do
  
  echo "Iteration", ${epoch}

  echo "Translating monolingual data..."
  
  # translate the monolingual data from random sample sentence (SRC to TGT)
  shuf -n ${SENT_BT} $SRC_TRUE > $SRC_TRUE.sample.${SENT_BT}.${epoch}.src
  $MOSES_BIN -threads $N_THREADS -f $TRAIN_DIR_ITER_FORWARD/model/moses.ini < $SRC_TRUE.sample.${SENT_BT}.${epoch}.src > $SRC_TRUE.sample.${SENT_BT}.${epoch}.tgt
  
  # train another model with direction (TGT to SRC)
  TRAIN_DIR_ITER_BACKWARD=${TRAIN_PATH}-${epoch}-backward
  mkdir -p $TRAIN_DIR_ITER_BACKWARD
  mkdir -p $TRAIN_DIR_ITER_BACKWARD/model
  mkdir -p $TRAIN_DIR_ITER_BACKWARD/corpus
  cp $SRC_TRUE.sample.${SENT_BT}.${epoch}.src $TRAIN_DIR_ITER_BACKWARD/corpus/src.sample.${SENT_BT}.${epoch}.src
  cp $SRC_TRUE.sample.${SENT_BT}.${epoch}.tgt $TRAIN_DIR_ITER_BACKWARD/corpus/src.sample.${SENT_BT}.${epoch}.tgt
  
  echo "Train Moses Backward"
  $TRAIN_MODEL -root-dir $TRAIN_DIR_ITER_BACKWARD -corpus ${TRAIN_DIR_ITER_BACKWARD}/corpus/src.sample.${SENT_BT}.${epoch} \
  -f tgt -e src -alignment grow-diag-final-and -reordering msd-bidirectional-fe \
  -lm 0:5:$SRC_LM_BLM:8 -external-bin-dir $UMT_PATH/moses/training-tools -mgiza \
  -cores $N_THREADS -max-phrase-length=4 -score-options "--NoLex" -first-step=1 -last-step=9
  
  echo "Removing lexical reordering features ..."
  mv $TRAIN_DIR_ITER_BACKWARD/model/moses.ini $TRAIN_DIR_ITER_BACKWARD/model/moses.ini.bkp
  cat $TRAIN_DIR_ITER_BACKWARD/model/moses.ini.bkp | grep -v LexicalReordering > $TRAIN_DIR_ITER_BACKWARD/model/moses.ini

  # translate the monolingual data from random sample 50m sentence (TGT to SRC)
  echo "Translating monolingual data..."
  shuf -n ${SENT_BT} $TGT_TRUE > $TGT_TRUE.sample.${SENT_BT}.${epoch}.tgt
  $MOSES_BIN -threads $N_THREADS -f $TRAIN_DIR_ITER_BACKWARD/model/moses.ini < $TGT_TRUE.sample.${SENT_BT}.${epoch}.tgt > $TGT_TRUE.sample.${SENT_BT}.${epoch}.src
  
  # train another model with direction (SRC to TGT)
  epoch_next=$((epoch + 1))
  TRAIN_DIR_ITER_FORWARD=${TRAIN_PATH}-${epoch_next}-forward
  mkdir -p $TRAIN_DIR_ITER_FORWARD
  mkdir -p $TRAIN_DIR_ITER_FORWARD/model
  mkdir -p $TRAIN_DIR_ITER_FORWARD/corpus
  cp $TGT_TRUE.sample.${SENT_BT}.${epoch}.src $TRAIN_DIR_ITER_FORWARD/corpus/tgt.sample.${SENT_BT}.${epoch}.src
  cp $TGT_TRUE.sample.${SENT_BT}.${epoch}.tgt $TRAIN_DIR_ITER_FORWARD/corpus/tgt.sample.${SENT_BT}.${epoch}.tgt
  
  echo "Train Moses Forward"
  $TRAIN_MODEL -root-dir $TRAIN_DIR_ITER_FORWARD -corpus $TRAIN_DIR_ITER_FORWARD/corpus/tgt.sample.${SENT_BT}.${epoch} \
  -f src -e tgt -alignment grow-diag-final-and -reordering msd-bidirectional-fe \
  -lm 0:5:$TGT_LM_BLM:8 -external-bin-dir $UMT_PATH/moses/training-tools -mgiza \
  -cores $N_THREADS -max-phrase-length=4 -score-options "--NoLex" -first-step=1 -last-step=9
  
  echo "Removing lexical reordering features ..."
  mv $TRAIN_DIR_ITER_FORWARD/model/moses.ini $TRAIN_DIR_ITER_FORWARD/model/moses.ini.bkp
  cat $TRAIN_DIR_ITER_FORWARD/model/moses.ini.bkp | grep -v LexicalReordering > $TRAIN_DIR_ITER_FORWARD/model/moses.ini

  echo "Translating test sentences..."
  $MOSES_BIN -threads $N_THREADS -f $TRAIN_DIR_ITER_FORWARD/model/moses.ini < $TRAIN_PATH/pro.true > $TRAIN_DIR_ITER_FORWARD/test.tgt.hyp.true
  echo "Detruecasing hypothesis..."
  $DETRUECASER < $TRAIN_DIR_ITER_FORWARD/test.tgt.hyp.true > $TRAIN_DIR_ITER_FORWARD/test.tgt.hyp.tok

done


echo "End of training. Experiment is stored in: $TRAIN_PATH"
