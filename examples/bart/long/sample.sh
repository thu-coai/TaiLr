DATA_DIR=$5
pt_file=checkpoint$2.pt
MODEL_DIR=$1


CUDA_VISIBLE_DEVICES=${4:0} fairseq-interactive ${DATA_DIR}-bin \
    --task translation \
    --path ${MODEL_DIR}/${pt_file} \
    --input $DATA_DIR/$3 \
    --nbest 1 \
    --beam 1 \
    --seed 42 \
    --buffer-size 500 --batch-size 16 \
    --source-lang src --target-lang tgt \
    --sampling \
    --sampling-topp 0.95 \
    --bpe gpt2 \
    --max-len-b 1024 \
    --remove-bpe | tee ${MODEL_DIR}/$3.gen.$2.log


grep ^D ${MODEL_DIR}/$3.gen.$2.log | cut -f3- > ${MODEL_DIR}/$3.gen.$2


