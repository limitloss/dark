![header](imgs/header.png)

<p align="center">
	<a href="CHANGELOG.md">
        <img src="https://img.shields.io/badge/version-1.0.0-green?style=flat-square&logo=appveyor.svg" alt="Version">
    </a>
    <a href="LICENSE">
        <img src="https://img.shields.io/badge/license-GPL--3.0-blue?style=flat-square&logo=appveyor.svg" alt="GPL-3.0 License">
    </a>
</p>

<br>



# DARK

Generate, score, and design massive numbers of hallucinated protein sequences with DARK models.

### About


In this first initial release we provide code and models for generating sequences from DARK models, calculating the IG-score of sequences, and predicting structure with Alphafold (v2.0). 

The models, code, and data here are described in the preprint *Design in the Dark: Learning Deep Generative Models for De Novo Protein Design* [on bioRxiv](https://www.biorxiv.org/content/10.1101/2022.01.27.478087v1) 
and *Using AlphaFold for Rapid and Accurate Fixed Backbone Protein Design* also [on bioRxiv](https://www.biorxiv.org/content/10.1101/2021.08.24.457549v1).
Some scripts and data are yet to be added, see the *To Do* section at the bottom of the README. 

**First dataset has been added! (7/Feb/2023)**

## Requirements

The following were used to develop DARK models and should be all you need: 
*   [python](https://www.python.org)      3.7
*   [pytorch](https://pytorch.org/)     1.7
*   [fast-transformers](https://github.com/idiap/fast-transformers)  0.4.0  (Easy install: `pip install --user pytorch-fast-transformers`)

There shouldn't be any issues with Python 3.5+ but this is untested. 
Code is tested and developed for Linux. 
For predicting structures using the provided AlphaFold script please see [the AlphaFold repo](https://github.com/deepmind/alphafold/) for installation details and our remarks below.

## Installation

Clone the directory

```sh
git clone https://github.com/limitloss/dark
cd dark
```

Now we download a tarball of the model weights from our google drive and then extract. 
The `wget` has extra steps to get around the google-drive big file warning.
This is downloading model parameters as `dark_params.tar.gz`, which is 1GB. 
We also download the background distribution which is a bit smaller (4MB) and doesn't need the work around.  

```sh
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1zIRHlZfWalhKnMl7olcbTDhPgczJBcmc' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1zIRHlZfWalhKnMl7olcbTDhPgczJBcmc" -O dark_params.tar.gz && rm -rf /tmp/cookies.txt
md5sum dark_params.tar.gz
tar -xvzf dark_params.tar.gz

wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1TNlkGbjv3SnRRFXQY7993OtIv95A9mU8' -O dark_bkgd.tar.gz
md5sum dark_bkgd.tar.gz
tar -xvzf dark_bkgd.tar.gz

``` 

The compressed directory file should have the following MD5 hash: `cae1df84d64e6500510abbd8899097ea`, using `md5sum`.
The hash for the background file is `aafbdf3655e5e0856c89e771ad861d7f`.
Uncompressed the params tar leaves you with a `params/` directory containing four different models, taking up about 1.1GB of space. 
If you have python, Pytorch, and fast-transformers installed you should be ready to go. 

## Downloading data

The first dataset is available covering over half a million hallucinated sequences.

First Data Release: 
 - 500000 Hallucinated Sequences, 100 residues long, the training set of the DARK3 model.
 - 950 example validation set and 950 example test set.

These are in `.txt` files with one sequence per line. 
I ripped them out of the original FASTA files to keep file size lower but I move it back soon. 
These links are very fiddly so if they break please do raise and issue or get in touch.


Also, very importantly, the test set and validation set here is extremely weakly split compared to the gold standard when working with natural protein sequences (splitting by structural classification).
The coming next release will have much better, though not perfect, train-test splits based on predicted structure classification. 

### The data 

Just `cd` into your DARK directory if you're not already and run the below to download the first dataset:
```sh
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1iYlM8498RFrAA2o3agqXbRsZWo58fqoi' -O dark_data.tar.gz
md5sum dark_data.tar.gz
tar -xvzf dark_data.tar.gz
```
The MD5 hash of the compressed directory is: `6e2c27387895e7e4b542e9e2794954a2`. This will give you a `data/` directory with a single (for now) `seq_basic/` subdirectory containing the sequences.
Additional data in the future will be added to separate sub directories here.

### Remaining data

**Coming Soon...:** The remaining datasets consisting of, for example, 100K AlphaFold2 structures (and an additional 1900 structures from the test and validation sets) and pLDDT predictions. 
There is also the same with variable length sequences, with different train-test splits as mentioned above, and different generation methods.


## Running DARK

The two key things you can do are generating sequences with dark models (`generate.py`) and scoring those sequences (`score.py`).
These two underpin the iterative process used by the DARK framework to progressively increase the size of the training set. 
We describe below how to use these command-line scripts.  

If instead you're looking for a better intuition of how the IG-score is calculated, please see the `calc_oracle()` method in `dark/score/igscore.py`.
The oracle model itself is in `dark/score/oracle.py`.

### Generating Sequences

Sequences from DARK models are generated with `generate.py`. Below is the output from running with the `--help` flag:

```sh
usr@machine:~$ python generate.py --help
usage: generate.py [-h] [-n n] [-d d] [-b b] samples

Generate sequences with DARK models

positional arguments:
  samples           Number of sequences samples (default: 1)

optional arguments:
  -h, --help        show this help message and exit
  -n n, --model n   Which model to use 1,2,or 3 (default: 3)
  -d d, --device d  Device to run on, Either: cpu or gpu (default: gpu)
  -b b, --batch b   batchsize for sampling (default: 1)

output --> stdout
```
Important to note this defaults to using a GPU (that your version of pytorch is happy with). 
If you'd like to run on CPU it's much slower, just change the `-d` flag to `cpu`. 
If you'd like to generate a large number of sequences you can increase the batch size for faster run time. 
This early version assumes the number of samples requested is evenly divisible by the batch size.
Below is a minimal example of running the model and generating 10 sequences which are written to a text file `test.txt`. 
A pre-generated version of the file is in the [example](./example/test.txt) directory.

```sh
usr@machine:~$ python generate.py 10 > test.txt
```
Each sequence is on a new line. Standard behavoir is for the output to be written to stdout. 

### Scoring Sequences

Generated sequences are scored with `score.py`, which prints the IG-score for each sequence (higher is better). 
Input is a text file with each sequence on a newline. 
Here are the outputs of running with the `--help` flag:
  
```sh
usr@machine:~$ python score.py --help
usage: score.py [-h] [-d d] [--stats] seqs

Calculate IG-score of sequences

positional arguments:
  seqs              Sequence file (*.txt file)

optional arguments:
  -h, --help        show this help message and exit
  -d d, --device d  Device to run on, Either: cpu or gpu (default: gpu)
  --stats           Calculate mean stats instead of per seq

output --> stdout
```

Usage in simplest case:
```sh
usr@machine:~$ python score.py test.txt
```
This prints the scores of each sequence onto a new line. The test.txt is the same as generated in the previous section.
If you'd like the mean and standard deviation of the scores of a sequences in a file then:
```sh
usr@machine:~$ python score.py test.txt --stats
```
The scoring function in this version assumes sequence length is 100 amino acids as in the work for simplicity.  


## Predicting Structure with AlphaFold

The script we provide for predicting structure of a single sequence (`alpha/predict_structure.py`) should be copied into the top level
AlphaFold directory to be used. Note, this has been tested and used with the earliest release versions of the AlphaFold2 repository. 
The extent to which it works effectively with the most current version of the repo is not yet tested. 
The following is the output with the `--help` flag:

```sh
usr@machine:~$ python predict_structure.py --help
usage: predict_structure.py [-h] seqfile outpath prefix

Predict structure of a single sequence with AlphaFold (v2)

positional arguments:
  seqfile     Sequence in a FASTA file (.fasta,.fas,.fa) or sequence file
              (.txt)
  outpath     Absolute path of desired output directory
  prefix      Output file prefix (default: seq, i.e. seq_relaxed.pdb
              seq_plddt.npy)

optional arguments:
  -h, --help  show this help message and exit

output --> refined structure (.pdb), per amino acid pLDDT (.npy)
```
Two files are output using the naming convention `<prefix>_relaxed.pdb` for the predicted structure and `<prefix>_plddt.npy` 
for the per-amino-acid pLDDT scores as a numpy binary. 
Here is an example of running the script with a FASTA file called `test.fas`: 
```sh
python predict_structure.py test.fas /out/dir/ seq
```
This writes two files to the `/out/dir/` directory, being `seq_relaxed.pdb` and `seq_plddt.npy`. 
The script by default predicts structures using all five AlphaFold models (the CASP14 models). 
Only the best predicted structure by average pLDDT score is relaxed with OpenMM and saved, being the aforementioned `.pdb` file and pLDDT file.

### Input sequence file

The input file can be a FASTA file or a sequence file, such as that produced by the `generate.py` script. The input file extension is used to judge file type.
If the input file contains several sequences then, by default, only the first sequence is used. 


## To Do:

I'll be quickly adding in (*hopefully*, @limitloss' thesis pemitting) the additional code for the remaining aspects of the project such as hallucinating/refining sequences, 
with either simulated annealing, backprop with a step-through operator, or both together.
Those are currently in the process of being refactored for release with the code contained herein.
We will also update with instructions, once we've sorted out hosting, for downloading the different datasets of hallucinated sequences that are used as training sets (as previously described above), along with any AlphaFold2 predicted structures, pLDDTs, etc.

- [ ] Test `predict_structure.py` in newest version of the AlphaFold2 repository.  
- [ ] Add greedy and MCMC fixed-backbone AlphaFold2 optimization script and AlphaFold2 hallucination script.  
- [ ] Add DMP2 gradient-based hallucination and greedy hill-climbing hallucination scripts, including refinement and dual options.
- [ ] Include updated links to the sequence, structure prediction, and pLDDT datasets for variable length and static length datasets. 
- [ ] Include variable length + topology conditional DARK model scripts with parameter downloads.    

## Changelog

For a log of recent changes please see the changelog in [CHANGELOG.md](./CHANGELOG.md). This is currently being updated manually by [@limitloss](https://github.com/limitloss).

## Citing

If you do use the contents of this repo in your work, we'd really appreciate a citation with [CITATION.bib](./CITATION.bib).

## Contributing

Although this repo is based on the work in [the preprints](https://www.biorxiv.org/content/10.1101/2022.01.27.478087v1), if you'd like to build on anything here, fix a bug, whatever it maybe, don't hesitate to contribute and submit a pull request!

## Contact

Current dev & maintainer is [@limitloss](https://github.com/limitloss). 

Please don't hesitate to reach out, either via:
- Email, which can be found by clicking the [link to the paper](https://github.com/psipred/s4pred/CHANGELOG.md), and clicking on Lewis' name. 
- Twitter [@limitloss](https://twitter.com/limitloss).  
- Mastodon [@lewis](https://fosstodon.org/@lewis)

[changelog]: ./CHANGELOG.md
[license]: ./LICENSE
[version-badge]: https://img.shields.io/badge/version-1.0.0-green?style=flat-square&logo=appveyor.svg
[license-badge]: https://img.shields.io/badge/license-GPL--3.0-blue?style=flat-square&logo=appveyor.svg



