SAVE_DIR=iwslt-mle-lr3e-4-ep80-ls0.1
EPOCHS=80

mkdir -p models/$SAVE_DIR

CUDA_VISIBLE_DEVICES=4 fairseq-train \
    data-bin/iwslt14.tokenized.de-en \
    --seed 42 \
    --save-dir models/$SAVE_DIR \
    --max-epoch $EPOCHS \
    --log-file models/$SAVE_DIR/log.txt \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 3e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric
