TASK=$1
MODEL_DIR=$2
cd ../../../
for SPLIT in train valid test
do
  for LANG in source target
  do
    python -m examples.roberta.multiprocessing_bpe_encoder \
    --encoder-json examples/bart/$MODEL_DIR/bpe/encoder.json \
    --vocab-bpe examples/bart/$MODEL_DIR/bpe/vocab.bpe \
    --inputs examples/bart/long/$TASK/$SPLIT.wp_$LANG \
    --outputs examples/bart/long/$TASK/$SPLIT.bpe.$LANG \
    --workers 20 \
    --keep-empty;
  done
done

cd examples/bart/long