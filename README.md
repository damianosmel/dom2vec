# dom2vec: Protein domain embeddings
Please note: repository in WIP, each folder indicated by WIP will be updated soon.
All protein domains analysis follows the data from ["Interpro"](https://www.ebi.ac.uk/interpro/) version 75.0.
All data associated with domains can be downloaded from the ftp site for this version, which can be found ["interpro/75.0"](ftp://ftp.ebi.ac.uk/pub/databases/interpro/75.0/).

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

## Build domain architecture
0. Data acquisition
For Interpro 75.0 version download the files:
* match_complete.xml.gz
* protein2ipr.dat.gz

1. Get protein lengths parsing match_complete.xml
* Change folder/files paths appropriately in 'code/proteinXMLHandler_run.py'
* run 'code/proteinXMLHandler_run.py'
* prot_id_len tabular will be created; a sample of the first 100 lines of the full file is saved at [](domain_architecture_creation/prot_id_len_sample_100.tab)

2. 

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
This repository is the implentation of the research paper: ["dom2vec: Unsupervised protein domain embeddings capture domains structure and function providing data-driven insights into collocations in domain architectures"](https://www.biorxiv.org/content/10.1101/2020.03.17.995498v2), currently in bioRxiv!
