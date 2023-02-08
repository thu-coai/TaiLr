TOTAL_NUM_UPDATES=100000
WARMUP_UPDATES=0
EPOCHS=5
LR=1e-4
MAX_TOKENS=8192
UPDATE_FREQ=1
BART_PATH=models/bart-base/model.pt
SAVE_DIR=bart-base-wp-tailr-min0.2-thres1e-7-lr1e-4-8192-ep5
DATA_PATH=data/writingPrompts-bin

mkdir -p models/$SAVE_DIR

cp $DATA_PATH/dict.src.txt $SAVE_DIR
cp $DATA_PATH/dict.tgt.txt $SAVE_DIR

CUDA_VISIBLE_DEVICES=6 fairseq-train $DATA_PATH \
    --seed 42 \
    --save-dir models/$SAVE_DIR \
    --log-format simple \
    --log-file models/$SAVE_DIR/log.txt \
    --tensorboard-logdir tb_log/$SAVE_DIR \
    --validate-interval 1 \
    --num-workers 20 \
    --restore-file $BART_PATH \
    --max-tokens $MAX_TOKENS \
    --task translation \
    --source-lang src --target-lang tgt \
    --truncate-source \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --arch bart_base \
    --criterion tailr \
    --label-smoothing 0.0 \
    --density-min-weight 0.2 \
    --density-ratio-threshold 1e-7 \
    --policy-update-per-k-epoch 1 \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.0 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --lr-scheduler fixed \
    --max-epoch $EPOCHS \
    --lr $LR --warmup-updates $WARMUP_UPDATES \
    --fp16 --update-freq $UPDATE_FREQ \
    --skip-invalid-size-inputs-valid-test \
    --find-unused-parameters;


