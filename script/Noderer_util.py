#!/usr/bin/env python
# coding: utf-8


import numpy as np
import sys, os, warnings, argparse,time
from icecream import ic
warnings.filterwarnings("ignore")




def Noderer_TIS_efficiency(frames):
    TIS_efficiency_nt_A = np.array([1.00,0.94,1.07,1.24,1.05,1.08,0,0,0,0.98,0.99])
    TIS_efficiency_nt_T = np.array([0.98,1.03,0.91,0.71,0.99,0.90,0,0,0,1.06,0.93])
    TIS_efficiency_nt_C = np.array([0.97,1.04,1.08,0.92,1.08,1.00,0,0,0,0.91,1.09])
    TIS_efficiency_nt_G = np.array([1.05,0.99,0.94,1.13,0.88,1.02,0,0,0,1.04,0.99])
    TIS_efficiency_nt_N = np.array([0   ,0   ,0   ,0   ,0   ,   0,0,0,0,   0,0   ])
    TIS_efficiency_PWM_normalize = np.array([TIS_efficiency_nt_A,TIS_efficiency_nt_T,TIS_efficiency_nt_C,TIS_efficiency_nt_G, TIS_efficiency_nt_N])
    TIS_efficiency_PWM = TIS_efficiency_PWM_normalize*76.9      # Translation efficiency values 的真值
    nt_pos = {'A':0, 'T':1, 'C': 2 , 'G': 3, 'N': 4}

    PWM_score = []
    for i in frames:
        score = 0
        for pos, char in enumerate(i):
            score += TIS_efficiency_PWM[nt_pos[char]][pos]
            #ic(TIS_efficiency_PWM[nt_pos[char]][pos])
        score_mean = score/8
        PWM_score.append(f'{score_mean:.2f}')
        #ic(f'{score_mean:.2f}')
    return np.array(PWM_score).reshape(-1, 1)


