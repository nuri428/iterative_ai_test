from glob import glob
import argparse
import pandas as pd
import os
from pathlib import Path
import random 
from sklearn.model_selection import train_test_split
random.seed = 0 

def main(args):
    
    preprocessed_df = pd.read_json(args.pre_data)
    train, test = train_test_split(preprocessed_df)
    if not os.path.exists(args.traindir):
        os.makedirs(args.traindir)
    train.to_json(f"{args.traindir}/train.json", orient='records')
    test.to_json(f"{args.traindir}/test.json", orient='records')


if __name__ == "__main__":
    argparse = argparse.ArgumentParser(description='data preprocessing')
    argparse.add_argument("pre_data", type=str, default="./data/sentiment_preprocessed/preprocessed.json", help="preprocessed dir")
    argparse.add_argument("traindir", type=str, default="./data/sentiment_train_data/", help="train preprocessing")
    args = argparse.parse_args()
    main(args)