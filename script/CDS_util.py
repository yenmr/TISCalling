
import numpy as np
from icecream import ic



stop_codons = ['TAG', 'TAA', 'TGA']
start_codon = 'ATG'

def length_to_first_stop_codon(seq, site):
    seq_len = len(seq)
    for i in range(site, seq_len, 3):
        codon = seq[i:i+3]
        if codon in stop_codons:
            end = i + 3
            length = end - site
            subseq = seq[site:end]
            gc_content = (subseq.count('G') + subseq.count('C')) / len(subseq) * 100
            return length, gc_content # +3 to include the stop codon itself
    end = seq_len
    length = end - site
    subseq = seq[site:end]
    gc_content = (subseq.count('G') + subseq.count('C')) / len(subseq) * 100

    return length, gc_content  # Return None if no stop codon is found

def upstream_content(seq, site):
    subseq = seq[:site]
    length = len(subseq)
    if length > 0:
        A_content = round(subseq.count('A') / length * 100, 1)
        C_content = round(subseq.count('C') / length * 100, 1)
        G_content = round(subseq.count('G') / length * 100, 1)
        T_content = round(subseq.count('T') / length * 100, 1)

        return A_content, C_content, G_content, T_content
    else:
        return 0, 0, 0, 0

def find_upstream_start_and_stop_codons(seq, site):
    seq_len = len(seq)
    upstream_start_codon_length = 200
    upstream_stop_codon_length = 200
    
    # Search for the closest upstream start codon
    for i in range(site - 3, -1, -1):
        codon = seq[i:i+3]
        #ic(i, codon, start_codon)
        if codon == start_codon:
            upstream_start_codon_length = site - i
            break
    
    # Search for the closest upstream stop codon
    for i in range(site - 3, -1, -1):
        codon = seq[i:i+3]
        
        #ic(i, codon, stop_codons)
        if codon in stop_codons:
            upstream_stop_codon_length = site - i
            break
    
    return upstream_start_codon_length, upstream_stop_codon_length

#2. Find CDS features
def pORF_finder(sites, seq):

    #Identify in-frame orf length
    pORF_length = []
    pORF_GC_content = []
    dis_stop_up = []
    dis_start_up = []
    pORF_up_A_content = []
    pORF_up_C_content = []
    pORF_up_G_content = []
    pORF_up_T_content = []
    for site in sites:
        orf_length, GC_content = length_to_first_stop_codon(seq, site)
        pORF_length.append(orf_length)
        pORF_GC_content.append(f'{GC_content:.2f}')
        #orf_length = length_to_first_stop_codon(seq, site)
        dis2start, dis2stop = find_upstream_start_and_stop_codons(seq, site)
        dis_start_up.append(dis2start)
        dis_stop_up.append(dis2stop)
        A_content, C_content, G_content, T_content = upstream_content(seq, site)
        pORF_up_A_content.append(A_content)
        pORF_up_C_content.append(C_content)
        pORF_up_G_content.append(G_content)
        pORF_up_T_content.append(T_content)
    features_name = ['corresponding_length_of_ORF', 
                     'corresponding_CG_content',
                     'upstream_corresponding_A_content', 
                     'upstream_corresponding_C_content',
                     'upstream_corresponding_G_content',
                     'upstream_corresponding_T_content', 
                     'dis_start_up', 
                     'dis_stop_up']

    features_value = np.array([pORF_length, pORF_GC_content, pORF_up_A_content, pORF_up_C_content, pORF_up_G_content, pORF_up_T_content, dis_start_up, dis_stop_up]).T
    #ic(features_value,features_name)
    return features_name, features_value
