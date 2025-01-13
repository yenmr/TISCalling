#!/usr/bin/env python
import os
import pandas as pd
import numpy as np
import argparse
import ml_util
from icecream import ic
#import json
import joblib

from Bio import SeqIO
import feature_util
import PWM_util
import time



def read_fasta(file_path):
    from Bio import SeqIO
    sequences = []
    with open(file_path, "r") as file:
        for record in SeqIO.parse(file, "fasta"):
            sequences.append({
                'id': record.id,
                'description': record.description,
                'sequence': str(record.seq)
            })
    return sequences

def find_start_codon(seq, codon_type):
    #seq = seqs[0]
    sequence = seq['sequence'].upper()
    sequence = ''.join([base if base in 'ACGT' else 'N' for base in sequence])
    seq_id = seq['id']
    # Define target codons
    target_codons = ["ATG"]
    if codon_type == 2:
        target_codons = target_codons + ["CTG", "TTG", "GTG", "AAG", "ACG", "AGG", "ATA", "ATC", "ATT"]
    
    # Find all positions of codon matches
    codon_positions = []
    codon_seq = []

    for i in range(0, len(sequence) - 2, 1):  # Iterate over the sequence in steps of 3
        codon = sequence[i:i+3]
        if codon in target_codons or codon_type ==3:
            codon_seq.append(codon)
            codon_positions.append(i)
    
    codon_positions_df = pd.DataFrame({'gene_ID':seq_id, 'codon': codon_seq, 'py_position':codon_positions})
    return codon_positions_df

def predict(features, model_path, classifier, site_id):
    model_dir = 'model'
    model_name = model_path.replace('Model_', '')
    sp = model_name.split('_')[0]

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
    #ic(TIS_codon_keep,features)
    target_scaler = f'{model_dir}/{model_path}/Feature_scaler.pkl'
    scaler = joblib.load(target_scaler)
    selected_features_name = scaler.get_feature_names_out().tolist()
    X_df_selected = features[selected_features_name]
    X_scaled = scaler.transform(X_df_selected)
    X_scaled_df = pd.DataFrame(X_scaled, columns=selected_features_name)

    #ic(outfile,new_df)
    #clf_names = ['LogisticRegression', 'RandomForest', 'GradientBoosting', 'SVM']
    clf_names = [classifier]
    #clf_names = ['GradientBoosting']
    for clf_name in clf_names:
        model_filename = f'{model_dir}/{model_path}/{clf_name}_model.pkl'
        best_classifier = joblib.load(model_filename)
        # Evaluate on the test set
        #y_test_pred = best_classifier.predict(X_scaled_df)
        y_test_pred_proba = best_classifier.predict_proba(X_scaled_df)[:, 1] if hasattr(best_classifier, "predict_proba") else None
        return y_test_pred_proba.round(2)

def main(args):

    Kmer_up = args.Kmer_up
    Kmer_down = args.Kmer_down
    RNA_win = args.RNA_win
    RNA_sli = args.RNA_sli
    RNA_border = args.RNA_border
    fasta_file = args.fasta
    codon_type = args.codon_type
    filter = args.filter
    model_file_path = args.models
    classifier = args.classifier

    ic(args)
    model_dir = "model"
    model_paths = model_file_path.split(",")

    clf_names = {'LR':'LogisticRegression', 'RF': 'RandomForest', 'GB': 'GradientBoosting', 'SVM': 'SVM'}
    clf_name = clf_names[classifier]
    seqs_dict = read_fasta(fasta_file)
    
    #only do first one
    for seq_dict in seqs_dict:
        #seq_dict = seqs_dict[0]
        start_time = time.time()
        seq = seq_dict['sequence'].upper()
        gene_id = seq_dict['id']
        group = find_start_codon(seq_dict, codon_type)
        #stop prediction if no sites
        if not len(group):
            continue

        group.sort_values(by='py_position', inplace=True)
        sites = group['py_position'].tolist()
        site_seq = PWM_util.gen_PWM_frames(sites, seq,left_offset=-3, right_offset=4)
        group['ID'] = group.apply(lambda row: f"{row['gene_ID']}|{row['py_position']+1}|{site_seq.pop(0)}", axis=1)
        #group['isATG'] = np.where(group['codon'] == 'ATG', 1, 0)
        sites_prepare_time = time.time() - start_time
        #print(f'Step 1 (Sites prepare): {sites_prepare_time:.6f} seconds')


        start_time = time.time()
        X_df = feature_util.feature_miner(sites, seq, Kmer_up=Kmer_up, Kmer_down=Kmer_down, RNA_win=RNA_win, RNA_sli=RNA_sli, RNA_border=RNA_border)
        X_df.index = group.index
        feature_miner_time = time.time() - start_time
        #print(f'Step 2 (Feature mining): {feature_miner_time:.6f} seconds')

        start_time = time.time()
        PWM_df = feature_util.pwm_miner(sites, seq)
        PWM_df.index = group.index
        TIS_codon_df = feature_util.TIS_codon_miner(sites, seq)
        TIS_codon_df.index = group.index

        X_combined_df = pd.concat([PWM_df, TIS_codon_df, X_df], axis = 1)
        pwm_miner_time = time.time() - start_time
        #print(f'Step 3 (PWM mining): {pwm_miner_time:.6f} seconds')

        #Create empty dataframe
        start_time = time.time()
        #predict_result = pd.DataFrame(index=group['py_position'])
        predict_result = pd.DataFrame(index=group['ID'])
        
        for model_path in model_paths:
            model_name = model_path.replace("Model_","")
            #ic(model_name, select_classifier.loc[model_name]['Classifier'])
            predict_result[model_name] = predict(X_combined_df, model_path, clf_name, group['ID'])
        pridiction_time = time.time() - start_time
        #print(f'Step 4 (Prediction): {pridiction_time:.6f} seconds')

        predict_result[predict_result < filter] = np.nan
        #ic(predict_result)
        outfile = f'Predict_{gene_id}.tsv'
        predict_result.to_csv(outfile, sep = "\t")

        predict_result_melt = pd.melt(predict_result.reset_index(), id_vars=['ID'], var_name='Model', value_name='Proba')
        predict_result_melt = predict_result_melt[predict_result_melt['Proba'] > filter].sort_values('ID').reset_index(drop=True)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Prepare input features table for learning model.')
    parser.add_argument('-fasta', required=True, help='FASTA file for prediction.')
    parser.add_argument('-models', required=True, help='Path to the model path. Separate by comma ",".')
    #parser.add_argument('-model', required=True, help='Path to the model path.')
    opt_group = parser.add_argument_group(title='OPTIONAL INPUT')
    opt_group.add_argument('-codon_type', help='What type of sites to predict. 1: ATG, 2: ATG and non-ATG, 3: all sites, default = 1',type=int, default=1)
    opt_group.add_argument('-filter', help='Prediction prabability cutoff. default = 0.5', type = float, default=0.5)
    opt_group.add_argument('-classifier', help='Classifier used for prediction. SVM, RF, LR, GB. default = SVM', type = str, default="SVM")
    #opt_group.add_argument('-logi_PWM', help='apply logistic function to limit the range of PWM score or not, default = False', default=False)
    opt_group.add_argument('-Kmer_up', help='length of upstream region used in calculating Kmer features, default = 99',type=int, default=99)
    opt_group.add_argument('-Kmer_down', help='length of downstream region used in calculating Kmer features, default = 99',type=int, default=99)
    opt_group.add_argument('-RNA_win', help='window size for RNA folding minimum free energy calculation, default=20',type=int, default=20)
    opt_group.add_argument('-RNA_sli', help='sliding size for RNA folding minimum free energy calculation, default=10',type=int, default=10)
    opt_group.add_argument('-RNA_border', help='sequence range: 5\'end,3\'end such as \'40,40\', default=\'40,40\'', default='40,40')

    args = parser.parse_args()
    main(args)

