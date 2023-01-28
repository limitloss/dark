#   !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
For predicting sequence structure using AlphaFold

After installing AlphaFold, copy this file into the top level AlphaFold directory.

See README.md for further details
"""
import sys
import os
import time
import numpy as np
import argparse
import warnings
from Bio import BiopythonWarning

# This silences Tensorflow dep. warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow 
tensorflow.compat.v1.logging.set_verbosity(tensorflow.compat.v1.logging.ERROR)

from alphafold.common import protein
from alphafold.data import pipeline
from alphafold.model import data
from alphafold.model import config
from alphafold.model import model
from alphafold.relax import relax



# Command line args
parser = argparse.ArgumentParser(description='Predict structure of a single sequence with AlphaFold (v2)', epilog='output --> refined structure (.pdb), per amino acid pLDDT (.npy)')
parser.add_argument('seq', metavar='seqfile', type=str,
                    help='Sequence in a FASTA file (.fasta,.fas,.fa) or sequence file (.txt)')
parser.add_argument('outpath', metavar='outpath', type=str,
                    help='Absolute path of desired output directory')
parser.add_argument('prefix', metavar='prefix', type=str, default='seq',
                    help='Output file prefix (default: seq, i.e. seq_relaxed.pdb seq_plddt.npy)')

args = parser.parse_args()
args_dict = vars(args)



#  Comment in the line below to use the cpu instead of gpu. About x10 slower
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# setup which model params to use
model_runners = {}
models = ["model_1", "model_2","model_3","model_4","model_5"]

# Set the same configs for all five models
for model_name in models:
    model_config = config.model_config(model_name)
    
    # Comment out below for standard random behavoir
    model_config.model.global_config.deterministic = True
    model_config.data.eval.feat.random_crop_to_size_seed = [42] # unnecessary but defensive
    
    # Turns off MSA resampling during recycling
    model_config.data.common.resample_msa_in_recycling=False

    model_config.data.common.num_recycle=0
    model_config.model.num_recycle=4 # This can be reduced for faster prediction but lower pLDDT
    model_config.data.eval.num_ensemble = 1
    model_config.model.resample_msa_in_recycling=False
    model_config.data.common.max_extra_msa=1
    
    # Template Off
    model_config.data.common.reduce_msa_clusters_by_max_templates=False
    model_config.data.common.use_templates=False
    model_config.model.embeddings_and_evoformer.template.embed_torsion_angles=False
    model_config.model.embeddings_and_evoformer.template.enabled=False
    
    # Zero weight heads are ignored and not calculated
    model_config.model.heads.predicted_aligned_error.weight=0
    model_config.model.heads.experimentally_resolved.weight=0
    model_config.model.heads.distogram.weight=0
    
    # MSAs are not used, and max clusters of 1 means a single sequence input
    model_config.data.eval.max_msa_clusters = 1
    model_config.data.eval.max_templates = 0
    model_config.data.eval.masked_msa_replace_fraction = 0
    
    model_params = data.get_model_haiku_params(model_name=model_name, data_dir=".")
    model_runner = model.RunModel(model_config, model_params)
    model_runners[model_name] = model_runner
  
def predict_structure(prefix, feature_dict, model_runners, random_seed=0, output_dir=''):  
    plddts = {}
    unrelaxed_pdbs = {}
    plddts_all={}
    for model_name, model_runner in model_runners.items():
        processed_feature_dict = model_runner.process_features(feature_dict, random_seed=random_seed)
        prediction_result = model_runner.predict(processed_feature_dict)
        
        unrelaxed_protein = protein.from_prediction(processed_feature_dict,prediction_result)
        unrelaxed_pdbs[model_name]=unrelaxed_protein
        
        plddts_all[model_name] = prediction_result['plddt']
        plddts[model_name] = np.mean(prediction_result['plddt'])
        
        print(f"{model_name} {plddts[model_name].mean()}")
                
    
    # Rank by pLDDT and write out only the relaxed PDB for the best prediction.
    ordering=sorted(plddts.items(), key=lambda x: x[1], reverse=True)
    model_name, _ = ordering[0]
        

    ranked_output_path = output_dir+f'{prefix}_relaxed.pdb'
    ranked_output_path_pl = output_dir+f'{prefix}_plddt.npy'
    
    savetime=time.time()
    
    np.save(ranked_output_path_pl,plddts_all[model_name])
    
    amber_relaxer = relax.AmberRelaxation(max_iterations=0,tolerance=2.39,
                              stiffness=10.0,exclude_residues=[],
                              max_outer_iterations=20)      
    relaxed_pdb_str, _, _ = amber_relaxer.process(prot=unrelaxed_pdbs[model_name])            
    with open(ranked_output_path, 'w') as f:
        f.write(relaxed_pdb_str)

    print("RELAX TIME: {:.3f}s".format(time.time()-savetime))
    return



def load_fasta(dataloc=''):
    sequences=[]
    counter=0
    scriptdir = os.path.dirname(os.path.realpath(__file__))
    
    # if it doesn't look like an absolute file path then we assume local
    if dataloc[0] != '/':
        path = scriptdir+'/'+dataloc
    else:
        path = dataloc
    
    with open(path,'r') as f:
        buffer=[]
        for idx, line in enumerate(f):
            line=line.rstrip()
            if line[0]=='>' and idx>0:
                sp=(counter, buffer[1])
                counter+=1
                # push to list
                sequences.append(sp)
                buffer=[]
                buffer.append(line)
            else:
                buffer.append(line)
            
    # Clear the final buffer
    sp=(counter, buffer[1])
    # push to list
    sequences.append(sp)
    print(f'Loaded {counter} Sequences')
    print('Using First Sequence Only')
    return sequences

def load_seq(dataloc=''):
    sequences=[]
    counter=0
    scriptdir = os.path.dirname(os.path.realpath(__file__))
    
    # if it doesn't look like an absolute file path then we assume local
    if dataloc[0] != '/':
        path = scriptdir+'/'+dataloc
    else:
        path = dataloc
    
    with open(path,'r') as f:
        for idx, line in enumerate(f):
            line=line.rstrip()
            sequences.append(line)
            counter+=1
    print(f'Loaded {counter} Sequences')
    print('Using First Sequence Only')
    return sequences    

# =============================================================================
# Main Logic
# =============================================================================






fileext=args.seq.split('.')[-1]
# check if FASTA format from file extension
if fileext in ('fa','fas','fasta'):
    # always take the first sequence in the file
    query_sequence = load_fasta(args.seq)[0][1]
else:
    query_sequence = load_seq(args.seq)[0]
    
outdir = args.outpath
prefix = args.prefix

# ensure formatting of output dir
if outdir[-1] != '/': outdir += '/'
    
with warnings.catch_warnings():
    warnings.simplefilter('ignore', BiopythonWarning)
    start = time.time()
    feature_dict = {
        **pipeline.make_sequence_features(sequence=query_sequence,
                                          description="none",
                                          num_res=len(query_sequence)),
        **pipeline.make_msa_features(msas=[[query_sequence]],
                                     deletion_matrices=[[[0]*len(query_sequence)]])
    }
    
    predict_structure(prefix,feature_dict,model_runners,output_dir=outdir)
    print("TIME: {:.3f}s".format(time.time()-start))
    sys.stdout.flush()


