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
    emotion_map = {-1:'neg',0:'neu',1:'pos'}
    preprocessed_df = df_all[['Index','RawText','GeneralPolarity']]    
    preprocessed_df.rename(columns={"Index":"id", "RawText":"sentence", "GeneralPolarity":"label"}, inplace=True)
    preprocessed_df = preprocessed_df[~preprocessed_df['label'].isna()]
    preprocessed_df['label'] = preprocessed_df['label'].map(emotion_map)
    
    # print(preprocessed_df.head())
    print(f"total data : {preprocessed_df.shape}")
    print(f"total labels : {preprocessed_df['label'].unique()}")
    if not os.path.exists(args.predir) :
        os.mkdir(args.predir)
    preprocessed_df.to_json(f"{args.predir}/datasets.json", orient='records')


if __name__ == "__main__":
    argparse = argparse.ArgumentParser(description='data preprocessing')
    argparse.add_argument("datadir", type=str, default="./data/sentiment_tagged_data", help="data preprocessing")
    argparse.add_argument("predir", type=str, default="./data/sentiment_train_data", help="preprocessed dir")
    args = argparse.parse_args()
    main(args)