DATA_DIR=$5
pt_file=checkpoint$2.pt
MODEL_DIR=$1


CUDA_VISIBLE_DEVICES=${4:0} fairseq-validate ${DATA_DIR}-bin \
    --task translation \
    --path ${MODEL_DIR}/${pt_file} \
    --valid-subset $3 \
    --batch-size 256 \
    --source-lang src --target-lang tgt \
    --results-path ${MODEL_DIR}/$3.lm.$2.res \
    --skip-invalid-size-inputs-valid-test | tee ${MODEL_DIR}/$3.lm.$2.log
    