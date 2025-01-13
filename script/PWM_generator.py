'''
Last updated: 2024.07.09
Compiler: Peter Lin
Usage: Obtaining species-specific PWM from genome sequences and ribosome profiling results
'''

# 0. Import nessasary modules and functions
import sys, os, warnings, argparse,time
import pandas as pd
import numpy as np
from tqdm import tqdm
#import logomaker
from icecream import ic
import math
warnings.filterwarnings("ignore")

# 1. Obtain user-defined arguments
parser = argparse.ArgumentParser(
    description= "Code for calculation of PWM for TP sites and TN sites.")

req_group = parser.add_argument_group(title='REQUIRED INPUT')
req_group.add_argument('-TP_TN_data',help='input TP and TN data',required=True)
req_group.add_argument('-seq',help='input sequence data',required=True)
req_group.add_argument('-TP_type', type=str, default='0', help='the type of positive sites used to calculate background information.\n 0: any, 1:Annotated, 2:5UTR_ATG, 3: CDS_ATG, 4: 5UTR_nonATG, 5: CDS_nonATG ')
req_group.add_argument('-filename',help='output file name',required=True)
opt_group = parser.add_argument_group(title='OPTIONAL INPUT')
opt_group.add_argument('-PWM_base', help='base for log calculation for PWM, default = np.e', default='np.e')
opt_group.add_argument('-min_freq',help='the minimum frequency of nucleotide at a certain position, default = 0.01', default=0.01)

if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

args = parser.parse_args()

# 2.1 Read TP, TN file
def read_TP_TN(path):
    TPTN = pd.read_csv(path, sep="\t",header=0)
    return TPTN

# 2.2 Read genome sequencing file
def read_seq(path):
    seq = pd.read_csv(path, sep="\t",names=['seq'], index_col=0)
    return seq

# 2.3 Fetch gene sequence around any positive site
def check_and_fetch_TP(TP, seq):
    check = []
    nTP = TP.shape[0]
    frames = []
    for i in range(nTP):
        geneID = TP.iloc[i,0]
        read = seq.loc[geneID].seq.upper()
        read = read.replace('W','N').replace('S','N').replace('M','N').replace('K','N').replace('R','N').replace('Y','N').replace('B','N').replace('D','N').replace('H','N').replace('V','N')
        site = TP.iloc[i,3]
        try:
            read = read[:site+10] # In case that the TIS site is discovered within last 10 nucleotides of the gene.
        except:
            N_add = 10 - len(read[site:]) # Add downstream N to the reads.
            read = read + 'N'*N_add
        codon_get = read[site:site+3]
        codon_need = TP.iloc[i,1]
        det = codon_get == codon_need
        if len(read) > 24:
            N_add = 0
            cut = site - 15
        else:
            N_add = 25 - len(read)
            cut = 0
        read = 'N'*N_add + read
        read = read[cut:]
        frames.append(read)
        check.append(det)
        if not det:
            print('\n' + geneID + ': The position of TIS is wrong!')
        else: 
            pass
    right = check.count(True)
    wrong = check.count(False)
    check_sum = '\nPositive set: There are ' + str(right) + ' sequences with correct TIS position and ' + str(wrong) + ' sequences with wrong TIS position.'
    print(check_sum)
    #print(frames)
    return frames

# 2.4 Fetch gene sequence around positive sites with annotations only 
def check_and_fetch_TP_annotated(TPTN, seq):
    TP_anno = TPTN.query('characteristics == "positive"')
    TP_anno = TP_anno.query('location == "Annotated"')
    check = []
    nTP = TP_anno.shape[0]
    frames = []
    for i in range(nTP):
        geneID = TP_anno.iloc[i,0]
        read = seq.loc[geneID].seq.upper()
        read = read.replace('W','N').replace('S','N').replace('M','N').replace('K','N').replace('R','N').replace('Y','N').replace('B','N').replace('D','N').replace('H','N').replace('V','N')
        site = TP_anno.iloc[i,3]
        read = read[:site+10]
        codon_get = read[site:site+3]
        codon_need = TP_anno.iloc[i,1]
        det = codon_get == codon_need
        if len(read) > 24:
            N_add = 0
            cut = site - 15
        else:
            N_add = 25 - len(read)
            cut = 0
        read = 'N'*N_add + read
        read = read[cut:]
        frames.append(read)
        check.append(det)
        if not det:
            print('\n'+ geneID + ': The position of TIS is wrong !')
        else: 
            pass
    right = check.count(True)
    wrong = check.count(False)
    check_sum = '\nPositive set: There are ' + str(right) + ' sequences with correct TIS position and ' + str(wrong) + ' sequences with wrong TIS position.'
    print(check_sum)
    #print(frames)
    return frames

# 2.5 Calculate the composition of nucleotides from all genomic sequences
def bg_nt_cal(cDNA):
    nA = 0
    nU = 0
    nC = 0
    nG = 0
    nN = 0
    nTotal = 0
    nSeq = cDNA.shape[0]
    for i in range(nSeq):
        seq = str(cDNA['seq'][i])
        nA += seq.count('A')
        nU += seq.count('T')
        nC += seq.count('C')
        nG += seq.count('G')
        nN += seq.count('N')
        nTotal += len(seq)
    ratio_A = nA/nTotal
    ratio_U = nU/nTotal
    ratio_C = nC/nTotal
    ratio_G = nG/nTotal
    ratio_N = nN/nTotal
    nt_profile = pd.DataFrame({'A': ratio_A,
                               'U': ratio_U,
                               'C': ratio_C,
                               'G': ratio_G,
                               'N': ratio_N},
                               index=['ratio'])
    if (nA + nU + nC + nG + nN) != nTotal:
        print('\nWARNING: There are ' + str(nTotal -(nA + nU + nC + nG + nN)) + ' unidentified nucleotides. Nucleotides can only be coded in A, T, C, G, N.')
    return nt_profile

# 3.1 Calculate the position frequency matrix using all sequences fetched
def PFM(RNA_window_list, status): #directly copied from 2.1_PWM_v2.py with slight modification
    if RNA_window_list == []:
        return None
    elif type(RNA_window_list) == type(None):
        return None
    else:
        # Create empty frequency_matrix[i][j] = 0
        # i=0,1,2,3,4 corresponds to A,T,C,G,N
        # j=0,...,length of dna_list[0]
        frequency_matrix = [[0 for v in RNA_window_list[0]] for x in 'ATCG']

        for dna in RNA_window_list:
            for index, base in enumerate(dna):
                if base == 'A':
                    frequency_matrix[0][index] +=1
                elif base == 'T':
                    frequency_matrix[1][index] +=1
                elif base == 'C':
                    frequency_matrix[2][index] +=1
                elif base == 'G':
                    frequency_matrix[3][index] +=1
                elif base == 'N':
                    pass
                else:
                    print('\nUnknown nucleotide symbol detected. The symbol is: ' + base)
        PFM_count = np.array(frequency_matrix)
        vertical_sum = PFM_count[0] + PFM_count[1] + PFM_count[2] +  PFM_count[3]
        PFM_percentage = PFM_count/vertical_sum
        print('\nThe calculation of PFM is completed...')
        #print(PFM_percentage)
        if status == "count":
            return PFM_count
        elif status == "percentage":
            return PFM_percentage

#PFM matrix could be stored as a dictionary obj: {'A':[...], 'T':[...]....}

# 3.2 Convert the position frenquency matrix into position weighted matrix
def PFM_to_PWM(PFM, bg_nt, log_base, min_freq):
    log_base = float(log_base)
    min_freq = float(min_freq)
    print('\nThe conversion of PFM to PWM is conducted with log base = ' + str(log_base) + ' and minimum frequency = ' + str(min_freq))
    freq_base_matrix = [[min_freq for v in range(25)] for x in 'ATCG']
    PFM = np.add(PFM, freq_base_matrix)
    
    PWM_A = np.emath.logn(log_base, np.divide(PFM[0], np.array(bg_nt.A)))
    PWM_T = np.emath.logn(log_base, np.divide(PFM[1], np.array(bg_nt.U)))
    PWM_C = np.emath.logn(log_base, np.divide(PFM[2], np.array(bg_nt.C)))
    PWM_G = np.emath.logn(log_base, np.divide(PFM[3], np.array(bg_nt.G)))
    PWM_N = np.array([0 for v in range(25)])

    PWM = {'A':PWM_A, 'T':PWM_T, 'C':PWM_C, 'G':PWM_G, 'N': PWM_N}
    print('\nThe calculation of PWM is completed...')

    return PWM

def PWM_gen():
    TPTN = read_TP_TN(args.TP_TN_data)
    seq = read_seq(args.seq)
    bg_nt = bg_nt_cal(seq)
    TP_type = args.TP_type
    TP = pd.DataFrame()
    if TP_type == '2' :
        TP = TPTN.query('characteristics == "positive" and location == "5UTR" and codon == "ATG"')
    elif TP_type == '3':
        TP = TPTN.query('characteristics == "positive" and location == "CDS" and codon == "ATG"')
    elif TP_type == '4':
        TP = TPTN.query('characteristics == "positive" and location == "5UTR" and codon != "ATG"')
    elif TP_type == '5':
        TP = TPTN.query('characteristics == "positive" and location == "CDS" and codon != "ATG"')
    elif TP_type == '1':
        TP = TPTN.query('characteristics == "positive" and location == "Annotated"')
    elif TP_type == '6':
        TP = TPTN.query('characteristics == "positive" and codon != "ATG"')
    elif TP_type == '7':
        TP = TPTN.query('characteristics == "positive" and codon == "ATG"')
    else:
        TP = TPTN.query('characteristics == "positive"')
    ic(TP_type,TP)

    frames = check_and_fetch_TP(TP=TP, seq=seq)

    #if TP_type == "any":
    #    frames = check_and_fetch_TP(TPTN=TPTN, seq=seq)
    #elif TP_type == "annotated":
    #    frames = check_and_fetch_TP_annotated(TPTN=TPTN, seq=seq)

    mat = PFM(frames, "percentage")

    if args.PWM_base == "np.e":
        input_base = np.e
    else:
        input_base = float(args.PWM_base)

    PWM = PFM_to_PWM(mat, bg_nt, input_base, float(args.min_freq))

    PWM = pd.DataFrame(PWM)
    
    PWM_dir = "PWM_library"
    os.makedirs(PWM_dir, exist_ok=True)
    outfile = os.path.join(PWM_dir, args.filename)
    PWM.to_csv(outfile, sep="\t",header=True,index=False)
    print('\nThe PWM is stored at ' + outfile)

PWM_gen()
