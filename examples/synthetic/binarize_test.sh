TASK=$1
fairseq-preprocess \
  --source-lang "src" \
  --target-lang "tgt" \
  --testpref "${TASK}/test" \
  --destdir "${TASK}-bin/" \
  --srcdict "data/coco-bin/dict.src.txt" \
  --tgtdict "data/coco-bin/dict.tgt.txt" \
  --workers 10 \