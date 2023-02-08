TOTAL_NUM_UPDATES=100000
EPOCHS=100
WARMUP_UPDATES=0
LR=1e-3
MAX_TOKENS=4096
UPDATE_FREQ=1
SAVE_DIR=coco_pseudo-mle-4096-lr1e-3-ep100
DATA_PATH=data/coco_pseudo-bin

mkdir -p models/$SAVE_DIR

CUDA_VISIBLE_DEVICES=6 fairseq-train $DATA_PATH \
    --seed 42 \
    --save-dir models/$SAVE_DIR \
    --log-file models/$SAVE_DIR/log.txt \
    --log-format simple \
    --tensorboard-logdir tb_log/$SAVE_DIR \
    --validate-interval 1 \
    --num-workers 10 \
    --max-tokens $MAX_TOKENS \
    --task translation \
    --source-lang src --target-lang tgt \
    --truncate-source \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --arch lstm \
    --encoder-embed-dim 128 \
    --encoder-hidden-size 128 \
    --encoder-layers 1 \
    --decoder-embed-dim 128 \
    --decoder-hidden-size 128 \
    --decoder-layers 1 \
    --decoder-out-embed-dim 128 \
    --decoder-attention 1 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.0 \
    --dropout 0.1 \
    --weight-decay 0.0 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --lr-scheduler fixed \
    --max-epoch $EPOCHS \
    --lr $LR --warmup-updates $WARMUP_UPDATES \
    --fp16 --update-freq $UPDATE_FREQ \
    --skip-invalid-size-inputs-valid-test \
    --find-unused-parameters;
