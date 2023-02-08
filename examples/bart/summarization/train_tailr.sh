TOTAL_NUM_UPDATES=200000
EPOCHS=5
WARMUP_UPDATES=0
LR=1e-4
MAX_TOKENS=8192
UPDATE_FREQ=1
BART_PATH=models/bart-base/model.pt
SAVE_DIR=bart-base-gigaword-tailr-min0.2-thres0.8-update1-ep5-ls0.1
DATA_PATH=data/gigaword-bin

mkdir -p models/$SAVE_DIR


CUDA_VISIBLE_DEVICES=6 fairseq-train $DATA_PATH \
    --seed 42 \
    --criterion tailr \
    --label-smoothing 0.1 \
    --density-min-weight 0.2 \
    --density-ratio-threshold 0.8 \
    --save-dir models/$SAVE_DIR \
    --log-file models/$SAVE_DIR/log.txt \
    --log-interval 100 \
    --log-format simple \
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
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.0 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --lr-scheduler fixed \
    --max-epoch $EPOCHS \
    --lr $LR --warmup-updates $WARMUP_UPDATES \
    --fp16 --update-freq $UPDATE_FREQ \
    --skip-invalid-size-inputs-valid-test \
    --find-unused-parameters;
