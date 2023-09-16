from glob import glob
import argparse
import json
import pandas as pd 
import os
from pathlib import Path 

def main(args):
    files = glob(f"{args.datadir}/**.json")
    df_list = []
    for file in files :
        df_list.append(pd.read_json(file))

    df_all = pd.concat(df_list)
    preprocessed_df = df_all[['Index','RawText','GeneralPolarity']]
    preprocessed_df = preprocessed_df.reset_index(names=["Index"])
    print(preprocessed_df.head())
    if not os.path.exists(args.predir) :
        os.mkdir(args.predir)
    preprocessed_df.to_json(f"{args.predir}/preprocessed.json")


if __name__ == "__main__":
    argparse = argparse.ArgumentParser(description='data preprocessing')
    argparse.add_argument("datadir", type=str, default="./data/sentiment_tagged_data", help="data preprocessing")
    argparse.add_argument("predir", type=str, default="./data/sentiment_preprocessed", help="preprocessed dir")
    args = argparse.parse_args()
    main(args)