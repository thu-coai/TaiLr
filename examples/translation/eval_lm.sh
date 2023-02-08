pt_file=checkpoint$2.pt
MODEL_DIR=$1


CUDA_VISIBLE_DEVICES=${4:0} fairseq-validate data-bin/iwslt14.tokenized.de-en \
    --task translation \
    --path ${MODEL_DIR}/${pt_file} \
    --valid-subset $3 \
    --batch-size 4 \
    --source-lang de --target-lang en \
    --results-path ${MODEL_DIR}/$3.lm.$2.res \
    --skip-invalid-size-inputs-valid-test \
    --remove-bpe | tee ${MODEL_DIR}/$3.lm.$2.log
    


#grep ^D ${MODEL_DIR}/$3.gen.log | cut -f3- > ${MODEL_DIR}/$3.gen.$2
