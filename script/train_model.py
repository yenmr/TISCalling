#!/usr/bin/env python
import os
import pandas as pd
import numpy as np
import argparse
import ml_util
from icecream import ic


def main(args):

    features_file_path = args.feature
    #features_file_path = '../1_Features/Features_Sl_5UTRATG.txt'
    feature_dir = "feature"
    features_file_path = os.path.join(feature_dir, features_file_path)
    features_file = os.path.basename(features_file_path)
    model_name = features_file.replace('.txt', '').replace('Features_', '')
    sp = model_name.split('_')[0]
    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)
    #out = 'Model_Sl_5UTRATG'

    features = pd.read_csv(features_file_path, sep = '\t')

    # Remove all columns with PWM except PWM_keep and rename to PWM
    PWM_keep = f'PWM_{model_name}'
    pwm_columns = [col for col in features.columns if 'PWM' in col]
    if PWM_keep in pwm_columns:
        columns_to_drop = [col for col in pwm_columns if col != PWM_keep]
        features = features.drop(columns=columns_to_drop)
        features = features.rename(columns={PWM_keep: 'PWM'})

    # Remove all comumns name with TIS_codon except TIS_codon_keep and rename to TIS_codon 
    TIS_codon_keep = f'TIS_codon_{model_name}'
    TIS_codon_columns = [col for col in features.columns if 'TIS_codon' in col]
    if PWM_keep in pwm_columns:
        columns_to_drop = [col for col in TIS_codon_columns if col != TIS_codon_keep]
        features = features.drop(columns=columns_to_drop)
        features = features.rename(columns={TIS_codon_keep: 'TIS_codon'})    

    # Decode tag and drop tag from features table
    tags = features['ID'].str.split('|', expand=True)
    tags.columns = ['Gene', 'Pos','Tag' ,'y']
    tags['y'] = tags['y'].astype(int)
    ic(tags['y'].value_counts())
    X_df = features.drop(columns='ID')
    y = np.array(tags['y'].tolist())
    ml_util.ml_training_and_testing(X_df, y, model_name, model_dir)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Prepare input features table for learning model.')
    parser.add_argument('-feature', required=True, help='Path to the feature table file.')

    args = parser.parse_args()
    main(args)

