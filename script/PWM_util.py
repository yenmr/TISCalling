import math 
import numpy as np

'''
def PWM_scorer(frames, PWM, logi):
    PWM_score = []
    for i in frames:
        score = 0
        pos = 0
        for j in i:
            score += PWM[j][pos]
            pos += 1
        if logi:
            score = (1/(1+math.exp(-score)) - 0.5)*2
        score = round(score,3)
        PWM_score.append(score)
    return PWM_score
'''
def PWM_scorer(frames, PWM, logi):
    PWM_score = []
    PWM_matrix = PWM.to_numpy()  # Convert the PWM DataFrame to a NumPy array for faster indexing
    column_index = {base: idx for idx, base in enumerate(PWM.columns)}

    for i in frames:
        indices = [column_index[base] for base in i]  # Get column indices for the current frame
        score = PWM_matrix[range(len(i)), indices].sum()  # Sum the scores using vectorized indexing
        
        if logi:
            score = (1 / (1 + math.exp(-score)) - 0.5) * 2
        
        score = round(score, 3)
        PWM_score.append(score)
    
    return PWM_score



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
