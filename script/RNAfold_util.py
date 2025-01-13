# The original file name is: generate_calculate_fas_RNAfold_TP-TN_data_transformed_v2.py. Renamed for simplicity.
##Genrate fasta formate for inoutting files to caluculate free energy (RNA structure)

#sys.argv[1]:TP_TN files; /nfs/data_C/MJL_ying/0_TN_TN_file/Tomato_TP_and_TN_data.txt; ##zero-based position
#sys.argv[2]:input:the seqeunces formate; for tomato cDNA seqeucnes; /nfs/data_C/MJL_ying/test_MJ/0_Data_Tomato/ITAG3.2_cDNA.fasta_seq.txt
#sys.argv[3]: window size, ex: 30
#sys.argv[4]: sliding size =15 or 1 (Lee-LTM, 2012, PNAS)

#sys.argv[5]: seqeucnes range [5ned, 3end] such as "60,60"--> upstream 60 nt and downstream

##output: (1)file_1: MFE in each window and log_calues : LOG2(|MEF|+1), ##the higher values means the lower of MEF at le and ri; (2)file_2: MFE in each window and variations : standard variations

####LogMEF-differ_left, LogMEF-differ_right: the positive/zero/negative vealues means the suceeding/preceeding window has higher/the same/lower MEF valaues
import sys, os, subprocess, numpy as np
from tqdm import tqdm
from icecream import ic
import ViennaRNA as vr
import pandas as pd
import argparse


    
'''
def get_mef(in_seq):
    #com_mef="/homes/mingliu/download_package/ViennaRNA/ViennaRNA-2.4.18/viennaRNA/bin/RNAfold"
    com_mef="/bcst/YAMADA/yenmr/miniconda3/envs/viennarna/bin/RNAfold"
    rna_mef = subprocess.Popen(com_mef, shell=True, stdin=subprocess.PIPE,stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    results, err = rna_mef.communicate(in_seq.encode())
    re_mef=float(results.decode().split("(")[-1].strip()[:-1])
    return re_mef
'''    

def get_mef(in_seq):
    (ss, mfe) = vr.fold(in_seq)
    #print("{}\n{} [ {:6.2f} ]".format(in_seq, ss, mfe))
    #re_mef = float(results.decode().split("(")[-1].strip()[:-1])
    re_mef = round(mfe, 3)
    #print(re_mef)
    return re_mef

def extract_subsequence(seq_ge, start, end):
    seq_len = len(seq_ge)
    
    # Initialize the subsequence with 'N' padding
    subseq = ['N'] * (end - start)
    
    # Calculate the actual indices within the original sequence
    for i in range(start, end):
        if 0 <= i < seq_len:
            subseq[i - start] = seq_ge[i]
    
    # Join the list into a string
    return ''.join(subseq)

def RNAfold(sites, seq, RNA_win, RNA_sli, RNA_border):  
    '''
    RNA_win = 20
    RNA_sli = 10
    RNA_border='40,40'
    '''

    #window size
    w_size=int(RNA_win)
    #sliding size
    sli_size=int(RNA_sli)

    ## load the seqeunces that indicated if any

    range5,range3=map(int, RNA_border.split(","))
    #print ("indicating sequence range", range5, range3)


    feature = []
    for i_s in range(0, range5, sli_size):
        #out.write("up_po:%s_win:%s\t"%(i_s, w_size))
        feature.append("up_po:%s_win:%s"%(i_s, w_size))
        
    for i_s in range(0, range3-10, sli_size):
        #out.write("down_po:%s_win:%s\t"%(i_s, w_size))
        feature.append("down_po:%s_win:%s"%(i_s, w_size))

    feature = feature + ["LogMEF-differ_left","LogMEF-differ_right","SD"]
    #print(feature)

    
    mefs = {}
    for f in feature:
        mefs[f] = []

    seq_ge = seq
    for site in sites:
    
        tis_po = site
        n = 0
        sd_v=[]
        le="NA"
        lele="NA"
        for i_s in range(0, range5, sli_size):
            start, end = tis_po-range5+i_s, tis_po-range5+i_s+w_size
            seq_out = extract_subsequence(seq_ge, start, end)
            '''
            if tis_po-range5+i_s>=0 and tis_po-range5+i_s+w_size <len(seq_ge):
                seq_out=seq_ge[(tis_po-range5+i_s):(tis_po-range5+i_s+w_size)]
            elif tis_po-range5+i_s<-w_size and tis_po-range5+i_s+w_size <len(seq_ge): #Use 20-N sequence as an alternative for complete empty region
                seq_out = 'N'*w_size
                #print('\n' + i_tis)
                #print('upstream ' + str(w_size) + 'NNNs!')
            elif tis_po-range5+i_s>=-w_size and tis_po-range5+i_s<0 and tis_po-range5+i_s+w_size <len(seq_ge): #Add corresponding Ns to make the sequence length = 20
                seq_out = seq_ge[0:(tis_po-range5+i_s+w_size)]
                seq_out = 'N'*abs(tis_po-range5+i_s) + seq_out
                #print('\n' + i_tis)
                #print('upstream pseudo-seqING')
            elif tis_po-range5+i_s>=0 and tis_po-range5+i_s+w_size>=len(seq_ge):
                seq_out=seq_ge[(tis_po-range5+i_s):]
                seq_out=seq_out + 'N'*(w_size - len(seq_out))
                #print('\n' + i_tis)
                #print('downstream pseudo-seqING')
                #ic(seq_out)
            else:
                #out.write("NA\t")
                #ic(extract_subsequence(seq_ge, start, end),seq_out)

                mefs[feature[n]].append("NA")
                n += 1
                sd_v.append("NA")
                continue
            '''
            #print i_tis, seq_out
            scre_mef=get_mef(seq_out)
            #out.write("%s\t"%(scre_mef))
            mefs[feature[n]].append(scre_mef)
            n += 1
            
            sd_v.append(scre_mef)
            
            if i_s == ((range5-1)//sli_size)*sli_size:
                le=scre_mef
            
            if i_s == (((range5-1)//sli_size)-1)*sli_size:
                lele=scre_mef
                

        ri="NA"
        riri="NA"
        for i_s in range(0, range3-sli_size, sli_size):
            start, end = tis_po+i_s, tis_po+i_s+w_size
            seq_out = extract_subsequence(seq_ge, start, end)
            '''
            if tis_po+i_s+w_size<len(seq_ge):
                seq_out=seq_ge[tis_po+i_s:(tis_po+i_s+w_size)]
                
            elif tis_po+i_s+w_size>=len(seq_ge) and tis_po+i_s<len(seq_ge):
                seq_out=seq_ge[tis_po+i_s:]
                seq_out = seq_out + 'N'*(w_size - len(seq_out))
                #print('\n' + i_tis)
                #print('downstream pseudo-seqING')
                #ic(extract_subsequence(seq_ge, start, end),seq_out)
            elif tis_po+i_s>=len(seq_ge):
                seq_out = 'N'*w_size
                #print('\n' + i_tis)
                #print('downstream ' + str(w_size) + 'NNNs!')
                #ic(extract_subsequence(seq_ge, start, end),seq_out)
            else:
                #out.write("NA\t")
                mefs[feature[n]].append("NA")
                n += 1
                sd_v.append("NA")
                continue
                #print i_tis, seq_out
            '''
            scre_mef=get_mef(seq_out)
            #out.write("%s\t"%(scre_mef))
            mefs[feature[n]].append(scre_mef)
            n += 1
                
            sd_v.append(scre_mef)
                                                
            if i_s == 0:
                ri=scre_mef
                
            if i_s == (0+1*sli_size):
                riri=scre_mef
                
                
        #print le, lele, ri, riri, sd_v
        
        if le =="NA" or lele =="NA":
            lo_le="NA"
        else:
            lo_le=np.log2(np.abs(le)+1)-(np.log2(np.abs(lele)+1)) ##the higher values means the lower of MEF at le
        
        if ri =="NA"or riri=="NA":
            lo_ri="NA"
        else:
            lo_ri=(np.log2(np.abs(ri)+1))-(np.log2(np.abs(riri)+1)) ##the higher values means the lower of MEF at le
        
        new_items = [item for item in sd_v if not isinstance(item, str)]
        std = np.std(new_items)
        
        #print le, lele, ri, riri, sd_v, new_items
                
        #out.write("%s\t%s\t%s\n"%(lo_le, lo_ri, std))
        #print(feature[n])
        mefs[feature[n]].append(f'{lo_le:.3f}')
        n += 1
        mefs[feature[n]].append(f'{lo_ri:.3f}')
        n += 1
        mefs[feature[n]].append(f'{std:.3f}')

    #ic(np.array( [mefs[fea] for fea in feature]).T)

    return feature, np.array( [mefs[fea] for fea in feature]).T



