split=${3:test}
CUDA_VISIBLE_DEVICES=${2:0} fairseq-generate data-bin/iwslt14.tokenized.de-en \
    --task translation \
    --source-lang de --target-lang en \
    --path $1/checkpoint_best.pt \
    --gen-subset $split \
    --batch-size 128 --beam 5 --remove-bpe | tee $1/$split.gen.log

grep ^H $1/$split.gen.log | cut -f3- > $1/$split.gen
grep ^T $1/$split.gen.log | cut -f2- > $1/$split.ref

fairseq-score -r $1/$split.ref -s $1/$split.gen > $1/$split.score