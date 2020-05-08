# dom2vec: Protein domain embeddings
Please note: repository in WIP, each folder indicated by WIP will be updated soon.

All protein domains analysis follows the data from [Interpro](https://www.ebi.ac.uk/interpro/) version 75.0.
All data associated with domains can be downloaded from the ftp site for this version, which can be found [ftp site](ftp://ftp.ebi.ac.uk/pub/databases/interpro/75.0/).

## Main dependencies
Code was executed using a conda environment, of which the full list of dependencies is in conda_env_dependencies.txt.

The main dependencies are listed below:
* Python 3.7.6
* BioPython 1.74
* Gensim 3.8.0
* Pytorch 1.2.0
* Torchtext 0.4.0
* Numpy 1.18.1
* Pandas 1.0.1
* Scikit-learn 0.22.1
* Matplotlib 3.1.1

## Build protein domain architectures
0. Data acquisition:

   For Interpro 75.0 version download the files:
* match_complete.xml.gz
* protein2ipr.dat.gz

1. Get protein lengths parsing match_complete.xml:
* Change folder/files paths appropriately in [proteinXMLHandler_run.py](code/proteinXMLHandler_run.py)
* run `proteinXMLHandler_run.py`
* prot_id_len tabular file will be created; a sample of the first 100 lines of the full file is saved at [sample file](domain_architecture_creation/prot_id_len_sample_100.tab)

2. Get domains and evidence db id per protein:
* Select the output domain annotation **type**: overlap, non overlapping or non redundant. Then set if *GAP* domain is also added to annotations. 
  Change folder/files paths appropropriately and uncomment the first section in [main.py](code/main.py) 
* Parse domain hits per protein running `main.py`
* id_domains_type.tab file will be created; a sample of the first 100 lines of the full file, for non overlapping with *GAP*, is saved at [sample file](domain_architecture_creation/id_domains_no_overlap_gap_sample_100.tab)

3. Get domain architecture corpus:
* Change folder/files paths appropriately and uncomment the first section in [main.py](code/main.py)
* run `main.py`
* domains_corpus_type.txt file will be created; sample of the first 100 line of the full file, for non overlapping with *GAP*, is saved at [sample file](domain_architecture_creation/domains_corpus_no_overlap_gap_sample_100.txt)

## Train protein domain embeddings
0. Needed data:
* the domains_corpus_type.txt from last step

1. Train word2vec model from domain architectures corpus:
* Change folder/files paths appropriately in [word2vec_run.py](code/word2vec_run.py)
* Change the paths and the training parameters in the provided bash script [run_embs.sh](domain_embeddings_training/run_embs.sh)
* run `run_embs.sh`
* word2vec embedding standard txt file(s) will be created

## Intrinsic evaluation - WIP
Data and example running experiments for:
* Domain hierarchy
* SCOPe and EC
* GO molecular function

## Downstream evaluation
Data and example running cross validation and performance experiments for three data sets:
* TargetP
* Toxin
* NEW

## Pretrained dom2vec - WIP

## Research paper
This repository is the implementation of the bioRxiv research paper:

["dom2vec: Unsupervised protein domain embeddings capture domains structure and function providing data-driven insights into collocations in domain architectures"](https://www.biorxiv.org/content/10.1101/2020.03.17.995498v2)
