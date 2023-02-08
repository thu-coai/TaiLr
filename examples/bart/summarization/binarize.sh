TASK=$1
MODEL_DIR=../models/bart-base/bpe
fairseq-preprocess \
  --source-lang "src" \
  --target-lang "tgt" \
  --trainpref "${TASK}/train.bpe" \
  --validpref "${TASK}/dev.bpe" \
  --testpref "${TASK}/test.bpe" \
  --destdir "${TASK}-bin/" \
  --workers 20 \
  --srcdict $MODEL_DIR/dict.txt \
  --tgtdict $MODEL_DIR/dict.txt;