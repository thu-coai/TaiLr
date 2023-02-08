import os
import subprocess
import argparse
import shutil

#root_dir = "data/coco_pseudo"
#train_num = 10000
#valid_num = 5000
#test_num = 5000
#model_dir = "coco-mle-4096-lr1e-3-ep50"
gpu_id = 0
seeds = ["42", "43", "44"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, help="root dir to save the data")
    parser.add_argument("--model_dir", type=str, help="path to the sampling model ckpts")
    parser.add_argument("--train_num", default=0, type=int, help="number of train data (value smaller than 1 means abandon this split)")
    parser.add_argument("--valid_num", default=0, type=int, help="number of valid data (value smaller than 1 means abandon this split)")
    parser.add_argument("--test_num", default=0, type=int, help="number of test data (value smaller than 1 means abandon this split)")
    parser.add_argument("--ckpt", default="_best", type=str, help="name of the ckpt to be used")
    parser.add_argument("--src_dict_dir", default="data/coco-bin/dict.src.txt", type=str, help="path to source dictionary")
    parser.add_argument("--tgt_dict_dir", default="data/coco-bin/dict.tgt.txt", type=str, help="path to target dictionary")
    args = parser.parse_args()

    os.makedirs(args.root_dir, exist_ok=True)
    os.makedirs(args.root_dir + "-bin", exist_ok=True)
    shutil.copy(args.src_dict_dir, args.root_dir + "-bin")
    shutil.copy(args.tgt_dict_dir, args.root_dir + "-bin")


    if args.train_num >= 1:
        with open(args.root_dir + "/train.src", "w") as f:
            for _ in range(args.train_num):
                f.write("<go>\n")

        print("Sample synth train data from the model...")
        subprocess.call(["bash", "generate.sh", args.model_dir, args.ckpt, "train.src", str(gpu_id), args.root_dir, seeds[0]])
        shutil.copyfile(args.model_dir + "/train.src.gen." + args.ckpt, args.root_dir + "/train.tgt")

    if args.valid_num >= 1:
        with open(args.root_dir + "/valid.src", "w") as f:
            for _ in range(args.valid_num):
                f.write("<go>\n")

        print("Sample synth valid data from the model...")
        subprocess.call(["bash", "generate.sh", args.model_dir, args.ckpt, "valid.src", str(gpu_id), args.root_dir, seeds[1]])
        shutil.copyfile(args.model_dir + "/valid.src.gen." + args.ckpt, args.root_dir + "/valid.tgt")

    if args.test_num >= 1:
        with open(args.root_dir + "/test.src", "w") as f:
            for _ in range(args.test_num):
                f.write("<go>\n")

        print("Sample synth test data from the model...")
        subprocess.call(["bash", "generate.sh", args.model_dir, args.ckpt, "test.src", str(gpu_id), args.root_dir, seeds[2]])
        shutil.copyfile(args.model_dir + "/test.src.gen." + args.ckpt, args.root_dir + "/test.tgt")


if __name__ == "__main__":
    main()