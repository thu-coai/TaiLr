MAX_LINE=10000
DEVICE=$2
head -n $MAX_LINE data/gigaword/dev.tgt > data/gigaword/dev.tgt.$MAX_LINE
head -n $MAX_LINE data/gigaword/dev.src > data/gigaword/dev.src.$MAX_LINE
for ep in {1..5}
do
    ./generate.sh $1 $ep dev.src.$MAX_LINE $DEVICE gigaword
    files2rouge data/gigaword/dev.tgt.$MAX_LINE $1/dev.src.$MAX_LINE.gen.$ep > $1/dev.score.$ep
done