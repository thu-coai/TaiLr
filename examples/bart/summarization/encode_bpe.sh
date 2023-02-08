DATA=$1
MODEL_DIR=$2
cd ../../../
for SPLIT in train dev test
do
  for LANG in src tgt
  do
    python -m examples.roberta.multiprocessing_bpe_encoder \
    --encoder-json examples/bart/$MODEL_DIR/bpe/encoder.json \
    --vocab-bpe examples/bart/$MODEL_DIR/bpe/vocab.bpe \
    --inputs examples/bart/summarization/$DATA/$SPLIT.$LANG.txt \
    --outputs examples/bart/summarization/$DATA/$SPLIT.bpe.$LANG \
    --workers 20 \
    --keep-empty;
  done
done

cd examples/bart/summarization