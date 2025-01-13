'''
Last updated: 2024.07.29
Compiler: Ming-Ren Yen
Usage: Generate all features from a single sequence
Notice: Please sort TP_TN_data by at least gene_ID before running the program.
Redesign the negative
Remove sites that the position less than 100
Add TIS_codon_usage features to reduce false positive

'''
# 0. import necessary modules
import sys, os, warnings, argparse, time
import pandas as pd
import numpy as np
from tqdm import tqdm
#import math
from icecream import ic
import PWM_util
#import Kmer_Kozak_util
#import Noderer_util
#import CDS_util
#import RNAfold_util
import feature_util

warnings.filterwarnings("ignore")
TIS_types = {'0':'All', '1':'Annotated', '2':'5UTRATG','3': 'CDSATG','4': '5UTRnonATG','5': 'CDSnonATG', '6':'nonATG', '7':'ATG'}



def main(args):
    '''
    TPTN = pd.read_csv('../0_TP_TN_file/Tomato_TP_and_TN_data.txt', sep="\t",header=0)
    seqs = pd.read_csv('../0_TP_TN_file/Tomato_seq.txt',sep="\t",names=['seq'], index_col=0)
    TIS_code = '2'
    TIS_sp = 'Sl'
    RNA_border = '40, 40'
    RNA_sli = 10
    RNA_win = 20
    Kmer_down = 99
    Kmer_up = 99
    TIS_code = '4'
    '''
    TP_TN_data = args.TP_TN_data
    seq_file = args.seq
    TIS_sp = args.TIS_sp
    Kmer_up = args.Kmer_up
    Kmer_down = args.Kmer_down
    RNA_win = args.RNA_win
    RNA_sli = args.RNA_sli
    RNA_border = args.RNA_border
    #negative_distance = args.negative_distance
    TIS_code = '0' if args.TIS_code not in ['1','2','3','4','5','6','7'] else args.TIS_code
    TPTN = pd.read_csv(TP_TN_data, sep = '\t', header=0)
    seqs = pd.read_csv(seq_file, sep = '\t', names=['seq'], index_col=0)

    TPTN_new = TPTN[TPTN['py_position'] >= 100]
    #TP = TPTN[(TPTN['characteristics'] == 'positive') & (TPTN['py_position'] > 100)]
    #TPTN_new = identify_negaitve(TP, seqs, 100)
    #TPTN_new.to_csv('../0_TN_TN_file/Tomato_TP_and_TN_new.txt', sep="\t",index=False)

    TIS_type = TIS_types[TIS_code]
    feature_dir = "feature"
    os.makedirs(feature_dir, exist_ok=True)
    out_file = os.path.join(feature_dir, f'Features_{TIS_sp}_{TIS_type}.txt')
    #out_file = f'Features_{TIS_sp}_{TIS_type}.txt'
    ic(args)
    ic(out_file, TIS_type)

    input_table = pd.DataFrame()
    if TIS_code == '2' :
        input_table = TPTN_new.query('location == "5UTR" and codon == "ATG"')
    elif TIS_code == '3':
        input_table = TPTN_new.query('location == "CDS" and codon == "ATG"')
    elif TIS_code == '4':
        input_table = TPTN_new.query('location == "5UTR" and codon != "ATG"')
        #input_table = select_near_negative(input_table, negative_distance)
    elif TIS_code == '5':
        input_table = TPTN_new.query('location == "CDS" and codon != "ATG"')
        #input_table = select_near_negative(input_table, negative_distance)
    elif TIS_code == '1':
        input_table = TPTN_new.query('(characteristics == "negative" or location == "Annotated") and codon == "ATG"')
        #input_table = select_near_negative(input_table, negative_distance)
    elif TIS_code == '6':
        input_table = TPTN_new.query('codon != "ATG"')
        #input_table = select_near_negative(input_table, negative_distance)
    elif TIS_code == '7':
        input_table = TPTN_new.query('codon == "ATG"')
    else:
        input_table = TPTN_new

    input_table.reset_index(drop=True,inplace=True)
    input_table.characteristics.value_counts()

    input_table['y'] = np.where(input_table['characteristics'] == 'positive', 1, 0)
    #input_table['ID'] = input_table.apply(
    #    lambda row: f"{row['gene_ID']}|{row['py_position']}|{row['y']}", axis=1
    #)
    ic(input_table.y.value_counts())
    X_list = []
    PWM_list = []
    TIS_codon_list = []
    grouped = input_table.groupby('gene_ID')
    tag_list = []
    #for gene_ID, group in grouped:
    for gene_ID, group in tqdm(grouped, dynamic_ncols=True):
        group.sort_values(by='py_position',inplace=True)
        sites = group['py_position'].tolist()
        seq = seqs.loc[gene_ID].seq.upper()
        frames = PWM_util.gen_PWM_frames(sites, seq,left_offset=-3, right_offset=4)
        #add TIS sequence to tag
        #group['ID'] = group.apply(lambda row: '|'.join(row['ID'].split('|')[:2] + [frames.pop(0)] + row['ID'].split('|')[2:]), axis=1)
        group['ID'] = group.apply(
            lambda row: f"{row['gene_ID']}|{row['py_position']}|{frames.pop(0)}|{row['y']}", axis=1
        )
        X = feature_util.feature_miner(sites, seq, Kmer_up=Kmer_up, Kmer_down=Kmer_down, RNA_win=RNA_win, RNA_sli=RNA_sli, RNA_border=RNA_border)
        X.index = group.index
        PWM = feature_util.pwm_miner(sites, seq)
        PWM.index = group.index
        TIS_codon_usage = feature_util.TIS_codon_miner(sites, seq)
        TIS_codon_usage.index = group.index
        X_list.append(X)
        PWM_list.append(PWM)
        TIS_codon_list.append(TIS_codon_usage)
        tag_list.append(group['ID'])

    #X_combined = pd.concat(X_list).reset_index(drop=True)
    #PWM_combined = pd.concat(PWM_list).reset_index(drop=True)
    #X_tag = pd.concat(tag_list).reset_index(drop=True)
    X_combined = pd.concat(X_list)
    PWM_combined = pd.concat(PWM_list)
    TIS_codon_combinded = pd.concat(TIS_codon_list)
    X_tag = pd.concat(tag_list)
    X_df = pd.concat([X_tag, PWM_combined, TIS_codon_combinded, X_combined], axis = 1)
    X_df.to_csv(out_file, index=False, sep = '\t')


if __name__ == '__main__':
    # 1. Obtain user-defined arguments
    parser = argparse.ArgumentParser(
        description= "Code for calculation of PWM score for TP sites and TN sites.")

    req_group = parser.add_argument_group(title='REQUIRED INPUT')
    req_group.add_argument('-TP_TN_data',help='input TP and TN data. MUST BE SORTED by "gene ID" before running the program',required=True)
    req_group.add_argument('-seq',help='input sequence data',required=True)
    req_group.add_argument('-TIS_sp', help='species to be analyzed. Sl for tomato, Hs for humans, Mm for mouse, At for arabidopsis.', required = True)
    #req_group.add_argument('-out', help='filename of the output. Remeber to specify the extension', required=True)
    req_group.add_argument('-TIS_code', type=str, help='the type of positive sites used to calculate background information.\n 0: All, 1:Annotated, 2:5UTR_ATG, 3: CDS_ATG, 4: 5UTR_nonATG, 5: CDS_nonATG',required=True)

    opt_group = parser.add_argument_group(title='OPTIONAL INPUT')
    opt_group.add_argument('-logi_PWM', help='apply logistic function to limit the range of PWM score or not, default = False', default=False)
    opt_group.add_argument('-Kmer_up', help='length of upstream region used in calculating Kmer features, default = 99',type=int, default=99)
    opt_group.add_argument('-Kmer_down', help='length of downstream region used in calculating Kmer features, default = 99',type=int, default=99)
    opt_group.add_argument('-RNA_win', help='window size for RNA folding minimum free energy calculation, default=20',type=int, default=20)
    opt_group.add_argument('-RNA_sli', help='sliding size for RNA folding minimum free energy calculation, default=10',type=int, default=10)
    opt_group.add_argument('-RNA_border', help='Sequence range: 5\'end,3\'end such as \'40,40\', default=\'40,40\'', default='40,40')
    #opt_group.add_argument('-negative_distance', help='Select the negative TIS range relative to positive TIS.',type=int, default=100)
    if len(sys.argv) == 1:
            parser.print_help()
            sys.exit(0)

    args = parser.parse_args()

    main(args)     



