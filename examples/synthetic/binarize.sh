TASK=$1
fairseq-preprocess \
  --source-lang "src" \
  --target-lang "tgt" \
  --trainpref "${TASK}/train" \
  --validpref "${TASK}/valid" \
  --destdir "${TASK}-bin/" \
  --workers 10 \
  ${@:2}
