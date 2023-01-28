#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scoring script for calculating IG-score of sequences

v1.0    Minimal Version;  

NOTE:   This is not a full implementation as it is assuming L=100. 

TO DO:  - Add dynamic length background calculation
        - Add option for input FASTA files
        - Print individual term scores of IG-score

Inputs (Required): .txt file, a text file with one sequence per line (e.g. output of generate.py)

See README.md for further details
"""

import os 
import argparse
import time
import torch

import numpy as np

from dark.score.igscore import igscorer


# Command line args
parser = argparse.ArgumentParser(description='Calculate IG-score of sequences', epilog='output --> stdout')
parser.add_argument('seqs', metavar='seqs', type=str,
                    help='Sequence file (*.txt file)')
parser.add_argument('-d','--device', metavar='d', type=str, default='gpu',
                    help='Device to run on, Either: cpu or gpu (default: gpu)')
parser.add_argument('--stats', dest='stats', action='store_true',
                    help='Calculate mean stats instead of per seq')
parser.set_defaults(stats=False)
args = parser.parse_args()
args_dict = vars(args)

    
sample_file = args.seqs #'test_100.txt'

bg_loc = 'bkgd/bg_100.npy'
oracle_loc = 'params/oracle.pt'

device = torch.device("cuda:0")

scriptdir = os.path.dirname(os.path.realpath(__file__))+'/'
bg_loc = scriptdir + bg_loc
oracle_loc = scriptdir + oracle_loc


igscore=igscorer(device, bg_loc=bg_loc, oracle_loc=oracle_loc)
    
# Load samples
samples=[]
with open(sample_file, 'r') as f:
    for line in f:
        samples.append(line.rstrip())
        
# print('Samples: {}'.format(len(samples)))




losses=[]
timing=[]
for idx, sample in enumerate(samples):
    start = time.time()
    
    loss = igscore.score(sample)
    losses.append(loss)
    timing.append(time.time()-start)
    

if args.stats:
    print(f'Prediction time: {np.mean(timing)} per seq')
    print(f"IG-score (mu) : {np.mean(losses)}")
    print(f"IG-score (sig): {np.std(losses)}")
else:
    for loss in losses:
        print(loss)


