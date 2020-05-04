from timeit import default_timer as timer
from Preprocess import Preprocess
from WordEmb import WordEmb
from Corpus import Corpus
from utils import sec2hour_min_sec

###                                      ###
###    Intepro domains --> embeddings    ###
###                                      ###

###                                 ###
### Setup Interpro Preprocess class ###
###                                 ###
"""
#Server run
data_path = "/home/damian/Documents/L3S/projects"
# data_path = "/data2/melidis" #server
prot_len_file_name = "prot_id_len.tab"

interpro_local_format = True #True for all data sets #False for protein2ipr.dat.gz (interpro_ftp)
#overlap -> with_overlap == True no matter the others
#no overlap -> with_overlap == False and with_redundant == False
#no_redundant -> with_overlap == False and with_redundant == True
with_overlap = False
with_redundant = False
with_gap = True
preprocess_protein2ipr = Preprocess(data_path, prot_len_file_name, with_overlap, with_redundant, with_gap, interpro_local_format)
"""

###                                ###
### Preprocess protein2ipr to      ###
### get domain architecture corpus ###
###                                ###
"""
# print("=====")
# print("1) Parsing protein2ipr -> protein_id tab domains")
### ###
# Processing Interpro data to learn embeddings from
### ###
#input: place the tabular.gz file in the data_path in order to proceed
#output: .tab file with protein id and their domains
#file_name = "protein2ipr.dat.gz" ##protein2ipr (interpro ftp)##
#file_name = "prot6.tab.gz" ##TEST##
batch_num_lines = 1000000
batch_out_prot = 10000
#credits: https://stackoverflow.com/questions/7370801/measure-time-elapsed-in-python
time_start = timer()
preprocess_protein2ipr.parse_prot2in(file_name, batch_num_lines, batch_out_prot)
time_end = timer()
print("Elapsed CPU time for parsing: {}.".format(sec2hour_min_sec(time_end-time_start)))

print("\n=====")
print("2) Parse id_domains.tab -> domains_corpus.txt")
file_in_name = "id_domains_no_overlap_gap.tab"#"id_domains_no_redundant_gap.tab"#"id_domains_overlap_gap.tab"
file_corpus_name = "domains_corpus_no_overlap_gap.txt"#"domains_corpus_overlap_gap.txt"#"domains_corpus_no_redundant_gap.txt"#
batch_num_lines = 100000
time_start = timer()
preprocess_protein2ipr.create_domains_corpus(file_in_name,file_corpus_name,batch_num_lines)
time_end = timer()
print("Elapsed CPU time for creating corpus: {}.".format(sec2hour_min_sec(time_end-time_start)))

print("\n=====")
print("3) Plot corpus histogram")
file_in = file_corpus_name#"domains_corpus_no_redundant_gap.txt"#"domains_corpus_no_overlap_gap.txt"#"domains_corpus_example.txt"#
domains_corpus = Corpus(data_path,file_in)
domains_corpus.plot_line_histogram()
"""

"""
###
# Train word2vec embeddings using corpus
###
print("\n=====")
print("4) Train domains_copurs.txt -> dom2vec.txt")
file_in = "domains_corpus_no_redundant_gap.txt"#file_corpus_name#"domains_corpus_no_overlap.txt"#"domains_corpus_prep1.txt"

#Train step-wise
### Word2vec Parameters ###
window = 10
use_cbow = 0
use_hierachical_soft_max = 0
vec_dim = 50
cores = 8
epochs_step = 5
max_epochs = 50
### ###

time_start = timer()
dom2Vec = WordEmb(data_path,file_in)
dom2Vec.set_up(window,use_cbow,use_hierachical_soft_max,vec_dim,cores)
dom2Vec.build_voc()
dom2Vec.train_stepwise(max_epochs,epochs_step)
time_end = timer()
print("Elapsed CPU time for initializing and training the model: {}.".format(sec2hour_min_sec(time_end-time_start)))
"""

###                                         ###
### Extract domains from proteins           ###
### in prediction data sets                 ###
"""
### ###
# Processing prediction data sets
# Get available domains for proteins
### ###
data_path = "/home/damian/Documents/L3S/projects"
prot_len_file_name = "prot_id_len.tab"

interpro_local_format = True #True for all data sets #False for protein2ipr.dat.gz (interpro_ftp)
#overlap -> with_overlap == True no matter the others
#no overlap -> with_overlap == False and with_redundant == False
#no_redundant -> with_overlap == False and with_redundant == True
with_overlap = True
with_redundant = False
with_gap = False
preprocess_domains4datasets = Preprocess(data_path, prot_len_file_name, with_overlap, with_redundant, with_gap, interpro_local_format)

print("======")
print("Prediction data sets A) map found domains to proteins of data set")
#file_name = "deeploc_remaining_seq.fasta_new.tsv.gz" ##DeepLoc##
file_name = "SP.715.rr.fasta.tsv.gz" #"nuc.1214.rr.fasta.tsv.gz" #"mTP.371.rr.fasta.tsv.gz" #"cyt.438.rr.fasta.tsv.gz" ##targetP##
# file_name = "targetp_remaining_seq_dataset_pos.fasta.tsv.gz" #"targetp_remaining_seq_dataset_pos.fasta.tsv.gz" #"targetp_remaining_seq_dataset_hard.fasta.tsv.gz" ##Toxin##
# file_name = "new_dataset_all.fasta.tsv.gz" ##NEW##
batch_num_lines = 1000000
batch_out_prot = 10000
#credits: https://stackoverflow.com/questions/7370801/measure-time-elapsed-in-python
time_start = timer()
preprocess_domains4datasets.parse_prot2in(file_name, batch_num_lines, batch_out_prot)
time_end = timer()
print("Elapsed CPU time for parsing: {}.".format(sec2hour_min_sec(time_end-time_start)))


### ###
# Processing prediction data sets
# Get unknown full-length domain for proteins without found domains
### ###
print("\n=====")
print("Prediction data sets B) parsing remaining fasta -> default domains tabular file")
## Input: fasta file and data_id_format
## all following fasta files should be placed in the data_path specified in the constructor of preprocess_domains4datasets
## Output: .tab file with 3 columns, protein ids, its domains and their evidence
## move this output to the respective dataset preprocessing subfolder

# fasta_name = "deeploc_remaining_seq2.fasta" ## DeepLoc ##
# fasta_name = "targetp_remaining_seq_Cytosole.fasta" #"targetp_remaining_seq_PathwaySignal.fasta" #"targetp_remaining_seq_Nuclear.fasta" #"targetp_remaining_seq_Mitochondrial.fasta" ##targetP##
# fasta_name = "targetp_remaining_seq_targetp_remaining_seq_dataset_pos.fasta" #"targetp_remaining_seq_targetp_remaining_seq_dataset_hard.fasta" #
# fasta_name = "new_remaining_seq.fasta" ## NEW ##
fasta_name = "targetp_remaining_seq_no_overlap.fasta" ## TargetP non overlapping ##
data_id_format = 1 #0 for DeepLoc and NEW #1 for TargetP #2 for Toxin
time_start = timer()
preprocess_domains4datasets.fasta2default_domains(fasta_name, data_id_format)
time_end = timer()
print("Elapsed CPU time for parsing: {}".format(sec2hour_min_sec(time_end-time_start)))
"""

print("=== * ===")
print("== *** ==")
