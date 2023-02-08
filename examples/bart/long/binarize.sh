TASK=$1
MODEL_DIR=../models/bart-base/bpe
fairseq-preprocess \
  --source-lang "source" \
  --target-lang "target" \
  --trainpref "${TASK}/train.bpe" \
  --validpref "${TASK}/valid.bpe" \
  --testpref "${TASK}/test.bpe" \
  --destdir "${TASK}-bin/" \
  --workers 20 \
  --srcdict $MODEL_DIR/dict.txt \
  --tgtdict $MODEL_DIR/dict.txt;