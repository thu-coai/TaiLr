import urllib.request
import os
import pickle
import subprocess

raw_data_url = {"train":"https://raw.githubusercontent.com/CR-Gjx/LeakGAN/master/Image%20COCO/save/realtrain_cotra.txt", 
                "test":"https://raw.githubusercontent.com/CR-Gjx/LeakGAN/master/Image%20COCO/save/realtest_coco.txt",
                "vocab":"https://raw.githubusercontent.com/CR-Gjx/LeakGAN/master/Image%20COCO/save/vocab_cotra.pkl"}

root_dir = "data/coco"

os.makedirs(root_dir, exist_ok=True)

if not os.path.isfile(root_dir + "/train.raw"):
    print("Downloading raw train data...")
    urllib.request.urlretrieve(raw_data_url["train"], root_dir + "/train.raw")
else:
    print("Raw train data exists!")
if not os.path.isfile(root_dir + "/test.raw"):
    print("Downloading raw test data...")
    urllib.request.urlretrieve(raw_data_url["test"], root_dir + "/test.raw")
else:
    print("Raw test data exists!")
if not os.path.isfile(root_dir + "/vocab.pkl"):
    print("Downloading vocab file...")
    urllib.request.urlretrieve(raw_data_url["vocab"], root_dir + "/vocab.pkl")
else:
    print("Vocab file exists!")

train_raw = [[int(y) for y in x.rstrip().split()] for x in open(root_dir + "/train.raw", "r").readlines()]
test_raw = [[int(y) for y in x.rstrip().split()] for x in open(root_dir + "/test.raw", "r").readlines()]
vocab, tok2idx = pickle.load(open(root_dir + "/vocab.pkl", "rb"))
idx2tok = dict([(v, k) for k, v in tok2idx.items()])

dev, train = train_raw[:5000], train_raw[5000:]

train_toks, dev_toks, test_toks = [], [], []
for line in train:
    toks = []
    for idx in line:
        toks.append(idx2tok[idx])
    train_toks.append(" ".join(toks))

for line in dev:
    toks = []
    for idx in line:
        toks.append(idx2tok[idx])
    dev_toks.append(" ".join(toks))


for line in test_raw:
    toks = []
    for idx in line:
        toks.append(idx2tok[idx])
    test_toks.append(" ".join(toks))

with open(root_dir + "/train.tgt", "w") as f:
    for line in train_toks:
        f.write(line + "\n")
with open(root_dir + "/train.src", "w") as f:
    for line in train_toks:
        f.write("<go>" + "\n")

with open(root_dir + "/test.tgt", "w") as f:
    for line in test_toks:
        f.write(line + "\n")
with open(root_dir + "/test.src", "w") as f:
    for line in test_toks:
        f.write("<go>" + "\n")

with open(root_dir + "/valid.tgt", "w") as f:
    for line in dev_toks:
        f.write(line + "\n")
with open(root_dir + "/valid.src", "w") as f:
    for line in dev_toks:
        f.write("<go>" + "\n")

print("Preprocessed data saved in data/coco")
subprocess.call(["bash", "binarize.sh", "data/coco"])
print("fairseq processed data saved in data/coco-bin")







