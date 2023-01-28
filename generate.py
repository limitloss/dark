#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generation script for sampling sequences from DARK Models

v1.0    Minimal Version;  

NOTE:   This is not a full implementation as it is assuming L=100. 

TO DO:  - Add batchsize flexibility
        - Add option for FASTA format output

Inputs (Required): int, number of desired sequences

See README.md for further details
"""
import argparse
import os 

import torch


from dark.model.model import DARK
from dark.model.data import seq2aa

# Command line args
parser = argparse.ArgumentParser(description='Generate sequences with DARK models', epilog='output --> stdout')
parser.add_argument('samples', metavar='samples', type=int, default=1,
                    help='Number of sequences samples (default: 1)')
parser.add_argument('-n','--model', metavar='n', type=str, default='3',
                    help='Which model to use 1,2,or 3 (default: 3)')
parser.add_argument('-d','--device', metavar='d', type=str, default='gpu',
                    help='Device to run on, Either: cpu or gpu (default: gpu)')
parser.add_argument('-b','--batch', metavar='b', type=int, default=1,
                    help='batchsize for sampling (default: 1)')

args = parser.parse_args()
args_dict = vars(args)
scriptdir = os.path.dirname(os.path.realpath(__file__))

# Load and freeze model
if args_dict['device'] == 'gpu': device = torch.device("cuda:0") 
else: device = torch.device("cpu:0") 

# start model and load model
if args.model == '1':
    dark=DARK(n_l=4, n_h=4, ff_d=128, q_d=32, v_d=32).to(device)
else:
    dark=DARK().to(device)
            
checkpoint = torch.load(scriptdir+'/params/dark'+args.model+'.pt')
dark.load_state_dict(checkpoint['model'])
dark.eval()


# sampling
with torch.no_grad():
    sample_num=args.samples
    loop_num=sample_num//args.batch

    for i in range(loop_num):
        samples = dark.generate(args.batch,device).cpu().tolist()
        for samp in samples:
            print(seq2aa(samp))

