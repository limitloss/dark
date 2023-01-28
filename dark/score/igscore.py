#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Class definition for calculating IG-score
"""

import torch
import numpy as np

from torch.nn.functional import one_hot

from dark.score.oracle import ORACLE


class igscorer:
    def __init__(self,device, bg_loc='', oracle_loc=''):
        
        self.device = device
        self.oracle = ORACLE(512,128).eval().to(device)
        self.oracle.eval()
        self.oracle.requires_grad_=False
        
        self.oracle.load_state_dict(torch.load(oracle_loc, map_location=lambda storage, loc: storage))         
        self.bg_dist = torch.tensor(np.load(bg_loc)).to(device)
        
        # p(x bkgd-pdb) from https://github.com/gjoni/trDesign 
        self.aa_bg = torch.tensor(np.array([0.07892653, 0.04979037, 0.0451488 , 0.0603382 , 0.01261332,
                                0.03783883, 0.06592534, 0.07122109, 0.02324815, 0.05647807,
                                0.09311339, 0.05980368, 0.02072943, 0.04145316, 0.04631926,
                                0.06123779, 0.0547427 , 0.01489194, 0.03705282, 0.0691271 ])).float().to(device)
        # for seq str --> int indices
        self.aa_trans = str.maketrans('ARNDCQEGHILKMFPSTWYVBJOUXZ-.', 'ABCDEFGHIJKLMNOPQRSTUUUUUUVV')
    
    def score(self,seq):
        length = len(seq)
        inputs = (np.frombuffer(''.join(seq).translate(self.aa_trans).encode('latin-1'), dtype=np.uint8) - ord('A')).reshape(1,length)
        inputs = torch.from_numpy(inputs).type(torch.LongTensor).to(self.device)
        loss = self.calc_oracle(inputs)
        return loss
    
    def calc_oracle(self,inputs,aa_weight=1.0):
        self.oracle.eval()
        with torch.no_grad():
            length = inputs.shape[1]
            msa1hot = one_hot(torch.clamp(inputs, max=20), 21).float()
            f2d_dca = torch.zeros((length, length, 442), device=self.device) 
            f2d_dca = f2d_dca.permute(2,0,1).unsqueeze(0)
            inputs2 = f2d_dca
            output = self.oracle(inputs, inputs2)
            output[0,0:2,:,:] = torch.softmax(output[0,0:2,:,:], dim=0)
            output[0,2:36,:,:] = torch.softmax(0.5 * (output[0,2:36,:,:] + output[0,2:36,:,:].transpose(-1,-2)), dim=0)
            output[0,36:70,:,:] = torch.softmax(output[0,36:70,:,:], dim=0)
            output[0,70:104,:,:] = torch.softmax(output[0,70:104,:,:], dim=0)
            
            # Four different distogram predictions
            ph = output[0,0:2,:,:]          # Presence of hydrogen bonds, 2D
            pd = output[0,2:36,:,:]         # Distances binned, 34D (0 is 0-4Ang, 1-32 is 0.5Ang increments, 33 is >=20Ang) 
            pph = output[0,36:70,:,:]       # Phi backbone torsion angle, 34D (equal bin width covering [0,2pi])
            pps = output[0,70:104,:,:]      # Psi backbone torsion angle, 34D (equal bin width covering [0,2pi])
            
            bkgd = self.bg_dist
            bkgd = bkgd.unsqueeze(0)
            bh = bkgd[0,0:2,:,:]
            bd = bkgd[0,2:36,:,:]
            bph = bkgd[0,36:70,:,:]
            bps = bkgd[0,70:104,:,:]
    
            # Calculate KL-Div for each of the four different distograms
            loss_hb = torch.mean(torch.sum(ph*torch.log(ph/bh),0))
            loss_dist = torch.mean(torch.sum(pd*torch.log(pd/bd),0))
            loss_phi = torch.mean(torch.sum(pph*torch.log(pph/bph),0))
            loss_psi = torch.mean(torch.sum(pps*torch.log(pps/bps),0))
                  
            # aa composition loss (we make this negative)
            aa_samp = torch.sum(msa1hot[0,:,:20],0)/msa1hot.size(1)+1e-7
            aa_samp = aa_samp/torch.sum(aa_samp)
            loss_aa = -torch.sum(aa_samp*torch.log(aa_samp/self.aa_bg))
    
            # total loss
            loss = loss_hb + loss_dist + loss_phi + loss_psi + aa_weight*loss_aa
    
            return loss.item()