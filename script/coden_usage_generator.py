'''
Last updated: 2024.07.09
Compiler: Peter Lin
Usage: Obtaining species-specific PWM from genome sequences and ribosome profiling results
'''

# 0. Import nessasary modules and functions
import sys, os, warnings, argparse,time
import pandas as pd
import numpy as np
#from tqdm import tqdm
#import logomaker
from icecream import ic
#import math
from collections import defaultdict
warnings.filterwarnings("ignore")

# 2.1 Read TP, TN file
def read_TP_TN(path):
    TPTN = pd.read_csv(path, sep="\t",header=0)
    return TPTN

# 2.2 Read genome sequencing file
def read_seq(path):
    seq = pd.read_csv(path, sep="\t",names=['seq'], index_col=0)
    return seq


def gen_PWM_frames(sites, seq, left_offset=-15, right_offset=10):
    subseqs = []
    seq_len = len(seq)
    seq = ''.join(['N' if base not in 'ACGT' else base for base in seq])
    for site in sites:
        start = site + left_offset
        end = site + right_offset

        # Initialize the subsequence with 'N' padding
        subseq = ['N'] * (right_offset - left_offset)

        # Calculate the actual indices within the original sequence
        for i in range(start, end):
            if 0 <= i < seq_len:
                subseq[i - start] = seq[i]

        subseqs.append(''.join(subseq))

    return subseqs


def count_codons(input_table, seqs):
    grouped = input_table.groupby('gene_ID')
    codon_counts = defaultdict(int)
    #for gene_ID, group in grouped:
    for gene_ID, group in grouped:
        group.sort_values(by='py_position',inplace=True)
        sites = group['py_position'].tolist()
        seq = seqs.loc[gene_ID].seq.upper()
        codons = gen_PWM_frames(sites, seq,left_offset=0, right_offset=3)
        for codon in codons:
            codon_counts[codon] += 1
    return codon_counts

def count_bg_codons(input_table, seqs):
    grouped = input_table.groupby('gene_ID')
    codon_counts = defaultdict(int)
    #for gene_ID, group in grouped:
    for gene_ID, group in grouped:
        seq = seqs.loc[gene_ID].seq.upper()
        coding_sequence = seq[group.iloc[0].CDS_start_py:group.iloc[0].CDS_end_py + 1]
        for i in range(0, len(coding_sequence), 3):
            codon = coding_sequence[i:i+3]
            codon_counts[codon] += 1

    return codon_counts

def count_bg_codons2(input_table, seqs):
    grouped = input_table.groupby('gene_ID')
    codon_counts = defaultdict(int)
    #for gene_ID, group in grouped:
    for gene_ID, group in grouped:
        seq = seqs.loc[gene_ID].seq.upper()
        #coding_sequence = seq[group.iloc[0].CDS_start_py:group.iloc[0].CDS_end_py + 1]
        for i in range(0, len(seq), 1):
            codon = coding_sequence[i:i+3]
            codon_counts[codon] += 1

    return codon_counts

def combine_codon_counts(codon_count1, codon_count2):
    # Create a new defaultdict to hold the combined counts
    combined_counts = defaultdict(int)

    for codon, count in codon_count1.items():
        combined_counts[codon] += count

    for codon, count in codon_count2.items():
        combined_counts[codon] += count

    return combined_counts

def compute_ratio(codon_count_TP, codon_count_TPTN):
    target_codons = ["ATG", "CTG", "TTG", "GTG", "AAG", "ACG", "AGG", "ATA", "ATC", "ATT"]

    codons_ratio = defaultdict(float)
    TP_sum = sum(codon_count_TP.values())
    TPTN_sum = sum(codon_count_TPTN.values())
    for codon, count_in_TP in codon_count_TP.items():
        #if codon in target_codons:
            ratio_TP = count_in_TP / TP_sum
            ratio_TPTN = codon_count_TPTN[codon] / TPTN_sum
            log2ratio = np.log2(ratio_TP / ratio_TPTN)
            codons_ratio[codon] = round(log2ratio, 3)
            #ic(codon,ratio_TP, ratio_TPTN, log2ratio)
    return codons_ratio


def main(args):
    TPTN = read_TP_TN(args.TP_TN_data)
    seqs = read_seq(args.seq)
    TP_type = args.TP_type
    sp = args.sp
    group = ''
    if TP_type == '2' :
        #TP = TPTN.query('characteristics == "positive" and location == "5UTR" and codon == "ATG"')
        TP = TPTN.query('characteristics == "positive" and location == "5UTR"')
        group = '5UTRATG'
    elif TP_type == '3':
        #TP = TPTN.query('characteristics == "positive" and location == "CDS" and codon == "ATG"')
        TP = TPTN.query('characteristics == "positive" and location == "CDS"')
        group = 'CDSATG'
    elif TP_type == '4':
        #TP = TPTN.query('characteristics == "positive" and location == "5UTR" and codon != "ATG"')
        TP = TPTN.query('characteristics == "positive" and location == "5UTR"')
        group = '5UTRnonATG'
    elif TP_type == '5':
        #TP = TPTN.query('characteristics == "positive" and location == "CDS" and codon != "ATG"')
        TP = TPTN.query('characteristics == "positive" and location == "CDS"')
        group = 'CDSnonATG'
    elif TP_type == '1':
        TP = TPTN.query('characteristics == "positive" and location == "Annotated"')
    elif TP_type == '6':
        TP = TPTN.query('characteristics == "positive" and codon != "ATG"')
    elif TP_type == '7':
        TP = TPTN.query('characteristics == "positive" and codon == "ATG"')
    else:
        TP = TPTN.query('characteristics == "positive"')
    ic(TP_type)


    #TP = TPTN.query('characteristics == "positive"')
    '''
    TPTN = read_TP_TN("../0_TP_TN_file/Tomato_TP_and_TN_data.txt")
    seqs = read_seq("../0_TP_TN_file/Tomato_seq.txt")
    TP_type = '4'
    '''

    codon_count_TP = count_codons(TP, seqs)
    codon_count_df = pd.DataFrame(list(codon_count_TP.items()), columns=['Codon', 'Value'])
    codon_count_df['Model'] = f'{sp}_{group}'
    total_count = sum(codon_count_TP.values())
    codon_percentage = {codon: round(np.log2((count / total_count) * 100 + 1),3) for codon, count in codon_count_TP.items()}
    codon_percentage_df = pd.DataFrame(list(codon_percentage.items()), columns=['Codon', 'Value']).sort_values(by='Value', ascending=False)

    codon_percentage_df['Model'] = f'{sp}_{group}'
    ic(codon_count_TP)
    codon_count_bg = count_bg_codons(TP, seqs)

    codon_ratio = compute_ratio(codon_count_TP, codon_count_bg)
    codon_ratio_df = pd.DataFrame(list(codon_ratio.items()), columns=['Codon', 'Value']).sort_values(by='Value', ascending=False)
    codon_ratio_df = codon_ratio_df[codon_ratio_df['Value'] > 0]
   
    codon_dir = "TIS_codon_library"
    os.makedirs(codon_dir, exist_ok=True)
    outfile = os.path.join(codon_dir, args.filename)

    codon_ratio_df.to_csv(outfile, sep="\t",header=True,index=False)
    print('\nThe TIS codon bias is stored at ' + outfile)


if __name__ == '__main__':
    # 1. Obtain user-defined arguments
    parser = argparse.ArgumentParser(
        description= "Code for calculation of PWM for TP sites and TN sites.")

    req_group = parser.add_argument_group(title='REQUIRED INPUT')
    req_group.add_argument('-TP_TN_data',help='input TP and TN data',required=True)
    req_group.add_argument('-seq',help='input sequence data',required=True)
    req_group.add_argument('-TP_type', type=str, default='0', help='the type of positive sites used to calculate background information.\n 0: any, 1:Annotated, 2:5UTR_ATG, 3: CDS_ATG, 4: 5UTR_nonATG, 5: CDS_nonATG ')
    req_group.add_argument('-filename',help='output file name',required=True)
    req_group.add_argument('-sp',help='species',required=True)
    if len(sys.argv) == 1:
            parser.print_help()
            sys.exit(0)

    args = parser.parse_args()
    main(args)