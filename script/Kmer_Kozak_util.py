#%%
### coding for K mers and complete/no Kozak element judgement
# Only binary features of Kmers were followed from PreTIS
# The frequency features of Kmers were normalized over the own length of the sequence
# there will be thoundsand of feature output to the file
# Py version: Python 3.11.4
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys, os, warnings, argparse, time
from icecream import ic
warnings.filterwarnings("ignore")
from itertools import chain

amino_acid_and_codon_dict={
    "Alanine" : ["GCT","GCC","GCA","GCG"]
    ,"Leucine" : ["TTA","TTG","CTT","CTC","CTA","CTG"]
    ,"Arginine" : ["CGT","CGC","CGA","CGG","AGA","AGG"]
    ,"Lysine" : ["AAA","AAG"]
    ,"Asparagine" : ["AAT","AAC"]
    ,"Methionine" : ["ATG"]
    ,"Aspartic_acid" : ["GAT","GAC"]
    ,"Phenylalanine" : ["TTT","TTC"]
    ,"Cysteine" : ["TGT","TGC"]
    ,"Proline" : ["CCT","CCC","CCA","CCG"]
    ,"Glutamine" : ["CAA","CAG"]
    ,"Serine" : ["TCT","TCC","TCA","TCG","AGT","AGC"]
    ,"Glutamic_acid" : ["GAA","GAG"]
    ,"Threonine" : ["ACT","ACC","ACA","ACG"]
    ,"Glycine" : ["GGT","GGC","GGA","GGG"]
    ,"Tryptophan" : ["TGG"]
    ,"Histidine" : ["CAT","CAC"]
    ,"Tyrosine" : ["TAT","TAC"]
    ,"Isoleucine" : ["ATT","ATC","ATA"]
    ,"Valine" : ["GTT","GTC","GTA","GTG"]
    ,"Stop_codon" : ["TAA","TGA","TAG"]
    ,"Shine_Dalgarno" : ["AGGAGG"]}

all_type_of_codon_list = list(chain.from_iterable([*amino_acid_and_codon_dict.values()]))


def gen_kmer_frames(sites, seq, left_offset=-99, right_offset=99, center = 3):
    subseqs = []
    seq_len = len(seq)
    seq = ''.join(['N' if base not in 'ACGT' else base for base in seq])
    for site in sites:
        start = site + left_offset
        end = site + right_offset

        # Initialize the subsequence with 'N' padding
        subseq = ['N'] * (right_offset - left_offset + center)

        # Calculate the actual indices within the original sequence
        for i in range(start, end + center):
            if 0 <= i < seq_len:
                subseq[i - start] = seq[i]

        subseqs.append(''.join(subseq))

    return subseqs

def is_kozake_complete(subseq, up):
    ## Kozake complete
    #ic(subseq[up-3:up+4])
    if (subseq[up-3] == 'A' or subseq[up-3] == 'G') and subseq[up+3] == 'G':
        #ic(subseq[up-3:up+4])
        return 1
    else:
        #ic(subseq[up-3:up+4])
        return 0

def is_no_kozake(subseq, up):
    ## No Kozake
    if (subseq[up-3] != 'A' or subseq[up-3] != 'G') and subseq[up+3] != 'G':
        #ic(subseq[up-3:up+4])
        return 1
    else:
        return 0

def is_nucleotide_position(subseq, up, down):
    ## K_mers_position_plus/minus_pos_is_NT
    feature_name = []
    feature_value = []
    for pos in range(-up, 0):
        for nt in 'ATCG': # Loop upstream area
            feature_name.append('K_mers_position_minus_' + str(abs(pos)) + '_is_' + nt)
            #feat_name = 'K_mers_position_minus_' + str(abs(pos)) + '_is_' + nt
            match = 1 if subseq[up + pos] == nt else 0
            #if(match == 1):
            #    ic(feat_name, match)
            feature_value.append(match)

    for pos in range(4, down + 1):
        for nt in 'ATCG': # Loop downstream area
            #feat_name = 'K_mers_position_plus_' + str(pos) + '_is_' + nt
            feature_name.append('K_mers_position_plus_' + str(pos) + '_is_' + nt)
            match = 1 if subseq[up + pos - 1] == nt else 0
            #if(match == 1):
            #    ic(feat_name, match)
            feature_value.append(match)
    return feature_name, feature_value

def count_nt_frequency(subseq, up, down):
    # frequency features:
    feature_name = []
    feature_value = []
    ## codon:
    ## all range 
    n_sites = len(subseq) // 3
    all_tri = n_sites*3 - 2
    for codon in all_type_of_codon_list:
        feat_name = 'K_mers_' + codon
        freq = subseq.count(codon)
        feature_name.append(feat_name)
        #feature_value.append(freq/all_tri)
        value = f'{freq/all_tri:.4f}'
        feature_value.append(value)

    ### codon_upstream: K_mers_upstream_codon
    n_sites = up // 3
    all_tri = n_sites*3 - 2
    for codon in all_type_of_codon_list:
        feat_name = 'K_mers_upstream_' + codon
        freq = subseq[:up].count(codon)
        feature_name.append(feat_name)
        #feature_value.append(freq/all_tri)
        value = f'{freq/all_tri:.4f}'
        feature_value.append(value)

    ### codon_downstream: K_mers_downstream_codon
    n_sites = down // 3
    all_tri = n_sites*3 - 2
    for codon in all_type_of_codon_list:
        feat_name = 'K_mers_downstream_' + codon
        freq = subseq[up+3:].count(codon)
        feature_name.append(feat_name)
        #feature_value.append(freq/all_tri)
        value = f'{freq/all_tri:.4f}'
        feature_value.append(value)

    ### in-frame_codon_downstream: K_mers_in_frame_downstream_codon
    n_sites = down // 3
    for codon in all_type_of_codon_list:
        feat_name = 'K_mers_in_frame_downstream_' + codon
        seq = subseq[up+3:]
        site = 0
        count = 0
        for g in range(len(seq)):
            site = seq.find(codon)
            det = site % 3
            if site == -1:
                break
            elif det == 0:
                count += 1
                seq = seq[site+3:]
                #print(seq)
            elif det != 0:
                seq = seq[site+(3-det):]
                #print(seq)
        feature_name.append(feat_name)
        #feature_value.append(count/n_sites)
        value = f'{count/n_sites:.4f}'
        feature_value.append(value)

    ### in-frame_codon_upstream: K_mers_in_frame_upstream_codon
    n_sites = up // 3
    for codon in all_type_of_codon_list:
        feat_name = 'K_mers_in_frame_upstream_' + codon
        seq = subseq[:up]
        seq = seq[::-1] # Reverse the sequence
        codon = codon[::-1] # Reverse the codon
        site = 0
        count = 0
        for g in range(len(seq)):
            site = seq.find(codon)
            det = site % 3
            if site == -1:
                break
            elif det == 0:
                count += 1
                seq = seq[site+3:]
                #print(seq)
            elif det != 0:
                seq = seq[site+(3-det):]
                #print(seq)
        feature_name.append(feat_name)
        #feature_value.append(count/n_sites)
        value = f'{count/n_sites:.4f}'
        feature_value.append(value)

    ## counts of NT:
    ### counts of NT in whole sequence: K_mers_nt
    full_len = len(subseq)
    for nt in 'ATCG':
        feat_name = 'K_mers_' + nt
        freq = subseq.count(nt)
        #feat_dict[feat_name] = freq/full_len
        feature_name.append(feat_name)
        #feature_value.append(freq/full_len)
        value = f'{freq/full_len:.4f}'
        feature_value.append(value)
 
    ### counts of upstream NT: K_mers_upstream_nt
    for nt in 'ATCG':
        feat_name = 'K_mers_upstream_' + nt
        freq = subseq[:up].count(nt)
        #feat_dict[feat_name] = freq/up
        feature_name.append(feat_name)
        #feature_value.append(freq/up)
        value = f'{freq/up:.4f}'
        feature_value.append(value) 

    ### counts of downstream NT: K_mers_downstream_nt
    for nt in 'ATCG':
        feat_name = 'K_mers_downstream_' + nt
        freq = subseq[up+3:].count(nt)
        #feat_dict[feat_name] = freq/down
        feature_name.append(feat_name)
        #feature_value.append(freq/down)
        value = f'{freq/down:.4f}'
        feature_value.append(value)
        
    return feature_name, feature_value

def count_aa_frequency(subseq, up, down):
    ## Counts of amino acid
    ### Counts of amino acid in whole sequence: K_mers_aa
    feature_name = []
    feature_value = []
    n_sites = len(subseq) // 3
    all_tri = n_sites*3 - 2
    for aa in amino_acid_and_codon_dict.keys():
        feat_name = 'K_mers_' + aa
        freq = 0
        for cd in amino_acid_and_codon_dict[aa]:
            freq += subseq.count(cd)
        #feat_dict[feat_name] = freq/all_tri
        feature_name.append(feat_name)
        #feature_value.append(freq/all_tri)
        value = f'{freq/all_tri:.4f}'
        feature_value.append(value)


    ### AA_upstream: K_mers_upstream_aa
    n_sites = up // 3
    all_tri = n_sites*3 - 2
    for aa in amino_acid_and_codon_dict.keys():
        feat_name = 'K_mers_upstream_' + aa
        freq = 0
        for cd in amino_acid_and_codon_dict[aa]:
            freq += subseq[:up].count(cd)
        #feat_dict[feat_name] = freq/all_tri
        feature_name.append(feat_name)
        #feature_value.append(freq/all_tri)
        value = f'{freq/all_tri:.4f}'
        feature_value.append(value)

    ### AA_downstream: K_mers_downstream_aa
    n_sites = down // 3
    all_tri = n_sites*3 - 2
    for aa in amino_acid_and_codon_dict.keys():
        feat_name = 'K_mers_downstream_' + aa
        freq = 0
        for cd in amino_acid_and_codon_dict[aa]:
            freq += subseq[up+3:].count(cd)
        #feat_dict[feat_name] = freq/all_tri
        feature_name.append(feat_name)
        #feature_value.append(freq/all_tri)
        value = f'{freq/all_tri:.4f}'
        feature_value.append(value)

    ### in-frame_AA_downstream: K_mers_in_frame_downstream_aa
    n_sites = down // 3
    for aa in amino_acid_and_codon_dict.keys():
        feat_name = 'K_mers_in_frame_downstream_' + aa
        count = 0
        for cd in amino_acid_and_codon_dict[aa]:
            seq = subseq[up+3:]
            site = 0
            for g in range(len(seq)):
                site = seq.find(cd)
                det = site % 3
                if site == -1:
                    break
                elif det == 0:
                    count += 1
                    seq = seq[site+3:]
                    #print(seq)
                elif det != 0:
                    seq = seq[site+(3-det):]
                    #print(seq)
        #feat_dict[feat_name] = count/n_sites
        feature_name.append(feat_name)
        #feature_value.append(count/n_sites)
        value = f'{count/n_sites:.4f}'
        feature_value.append(value)

    ### in-frame_AA_upstream: K_mers_in_frame_upstream_aa
    n_sites = up // 3
    for aa in amino_acid_and_codon_dict.keys():
        feat_name = 'K_mers_in_frame_upstream_' + aa
        count = 0
        for cd in amino_acid_and_codon_dict[aa]:
            seq = subseq[:up]
            seq = seq[::-1] # Reverse the sequence
            cd = cd[::-1] # Reverse the codon
            site = 0
            for g in range(len(seq)):
                site = seq.find(cd)
                det = site % 3
                if site == -1:
                    break
                elif det == 0:
                    count += 1
                    seq = seq[site+3:]
                    #print(seq)
                elif det != 0:
                    seq = seq[site+(3-det):]
                    #print(seq)
        #feat_dict[feat_name] = count/n_sites
        feature_name.append(feat_name)
        #feature_value.append(count/n_sites)
        value = f'{count/n_sites:.4f}'
        feature_value.append(value)
    return feature_name, feature_value

def kmer_miner(sites, seq, up = 99, down = 99):
    subseqs = gen_kmer_frames(sites, seq, left_offset = -up, right_offset = down)
    features_name =[]
    features_value = []
    for subseq in subseqs:
        kz_features_value = []
        kz_features_name = []
        kz_features_value.append(is_kozake_complete(subseq, up))
        kz_features_name.append('Complete_Kozake')
        kz_features_value.append(is_no_kozake(subseq, up))
        kz_features_name.append('No_Kozake')

        is_nt_feature_name, is_nt_feature_value = is_nucleotide_position(subseq, up, down)
        count_nt_feature_name, count_nt_feature_value = count_nt_frequency(subseq, up, down)
        count_aa_feature_name, count_aa_feature_value = count_aa_frequency(subseq, up, down)
        features_name = kz_features_name + is_nt_feature_name + count_nt_feature_name + count_aa_feature_name
        features_value.append(kz_features_value + is_nt_feature_value + count_nt_feature_value + count_aa_feature_value)
    return features_name, np.array(features_value)
