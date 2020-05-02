from os.path import join
from PrepareDataSet import PrepareDataSet
from DeepLocExperiment import DeepLocExperiment
from TargetPExperiment import TargetPExperiment
from ToxinExperiment import ToxinExperiment
from NEWExperiment import NEWExperiment

###                    ###
### DeepLoc Experiment ###
###                    ###
"""
###
# from input fasta to DeepLoc data with sequence and domains and label columns
###
print(" === DeepLoc Experiment ===")
# from deeploc_data.fasta -> data set csv#
fasta_path = "/home/damian/Documents/L3S/projects/datasets/deeploc/deeploc_data.fasta"
domains_path = "/home/damian/Documents/L3S/projects/no_red_gap/id_domains_no_redundant_gap.tab"
output_path = "/home/damian/Documents/L3S/projects/datasets/deeploc"

# For deeploc_remaining_seq.fasta
# fasta_path = "/home/damian/Documents/L3S/projects/datasets/deeploc/deeploc_remaining_seq.fasta"
# domains_path = "/home/damian/Documents/L3S/projects/datasets/deeploc/id_domains_no_redundant_gap_remaining_seq.tab"
fasta_path = "/home/damian/Documents/L3S/projects/datasets/deeploc/deeploc_remaining_seq2.fasta"
domains_path = "/home/damian/Documents/L3S/projects/datasets/deeploc/default_domains.tab"
output_path = "/home/damian/Documents/L3S/projects/datasets/deeploc"

label_name = "cellular_location"#"membrane_soluble"#
output_path = join(output_path, label_name)
DeepLocExperiment = DeepLocExperiment(fasta_path, domains_path, output_path, label_name)
print("1) deeploc_data.fasta -> dataset csv.")
#location: get all proteins
value2remove = ""
# value2remove = "U" #membrane: remove proteins with unknown assignment
data_set_path = DeepLocExperiment.fasta2csv(value2remove)
"""

###
# create DeepLoc location and membrane data set
###
"""
#=== remove duplicate, shuffle split train/test & k-fold split ===#
#place initial dataset.csv to the output folder then the train,test and cross-validation sets will be created.
print("=== ~ ===")
print("2) dataset.csv -> dataset_train.csv + dataset_test.csv")
output_path = "/home/damian/Documents/L3S/projects/datasets/deeploc"
label_name = "membrane_soluble" #"cellular_location" #
data_set_path = join(output_path, label_name)#when you have many csv files
#data_set_path = join(output_path,label_name)#when you have one csv file
output_path = join(output_path, label_name)
dataset_name = "deeploc_" + "dataset_" + label_name
PrepareDataSet = PrepareDataSet(data_set_path, output_path, dataset_name)
used_columns = ["train_test", "cellular_location", "membrane_soluble", "seq", "interpro_domains"]
PrepareDataSet.read_dataset(used_columns)

# clean from non Interpro domains
id_col_exists = False
PrepareDataSet.remove_no_interpro_domains(used_columns, id_col_exists)
# remove duplicate instances
PrepareDataSet.remove_duplicate_instances("interpro_domains", None)
PrepareDataSet.remove_duplicate_instances("seq", None)

# check the number of protein-length unknown domains
PrepareDataSet.count_num_proteins_per_dom_type()
# remove proteins with no interpro domains
PrepareDataSet.remove_unk_gap_proteins()
# shuffle data set
PrepareDataSet.shuffle_dataset()
remove_ids = False
PrepareDataSet.save_dataset(remove_ids)

## split train/test stratified on y ##
print("=====")
print("3.1) split train/test stratified on label")
test_portion = 0.3
PrepareDataSet.split_train_test_stratified(used_columns, label_name, test_portion)

## check the percentage of OOV for domains between train and test ##
print("=====")
print("3.2) Check domains distribution in train and test")
output_path = "/home/damian/Documents/L3S/projects/datasets/deeploc"
data_set_path = join(output_path, label_name)
output_path = join(output_path, label_name)
dataset_name = "deeploc_dataset_" + label_name
# PrepareDataSet = PrepareDataSet(data_set_path, output_path, dataset_name)
train_name = "deeploc_dataset_" + label_name + "_train.csv"
test_name = "deeploc_dataset_" + label_name + "_test.csv"
use_test4analysis = True
unk_domains_exist = False
PrepareDataSet.diagnose_oov_domains(train_name, test_name, use_test4analysis, unk_domains_exist)

## inner fold cross validation stratified on y ##
print("=====")
print("3.3) create inner 3 fold cross validation sets")
if label_name == "cellular_location":
	x_columns = ["train_test", "cellular_location", "seq", "interpro_domains"]
else:
	x_columns = ["train_test", "membrane_soluble", "seq", "interpro_domains"]

k = 3
PrepareDataSet.split_inner_stratified_kfold(k, x_columns, label_name)

## subsample data with different percentages ##
print("=====")
print("3.4) subsample train set")
percentages, num_picks = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5], 20
PrepareDataSet.subset_train(percentages, num_picks)

print("=====")
print("4) split test per OOV")

# if you create the OOV splits individually after creation of the data set
# define the input, output
# output_path = "/home/damian/Documents/L3S/projects/datasets/deeploc"
# label_name = "cellular_location"
# label_name = "membrane_soluble"
# dataset_name = "deeploc_" + "dataset_" + label_name
# output_path = join(output_path, label_name)
# data_set_path = output_path
# PrepareDataSet = PrepareDataSet(data_set_path, output_path, dataset_name)
# used_columns = ["train_test", "cellular_location", "membrane_soluble", "seq", "interpro_domains"]

train_file = "deeploc_dataset_" + label_name + "_train.csv"
test_file = "deeploc_dataset_" + label_name + "_test.csv"

oov_percentages = [0, 0.1, 0.3, 0.5, 0.7, 1.0]
PrepareDataSet.split_test_per_oov_percentage(train_file, test_file, oov_percentages, used_columns)
print("=== ~ ===")
"""

###                    ###
### TargetP Experiment ###
###                    ###
"""
###
# from input fasta to TargetP data with sequence and domains and label columns
###
print("=== TargetP Experiment ===")
print("1) 4 classes fasta -> dataset csv.")
#step A. mapping fasta files to interpro version 75.0
# fasta_dir = "/home/damian/Documents/L3S/projects/datasets/targetp/fasta_set"
# domains_path = "/home/damian/Documents/L3S/projects/linear/no_red_gap/id_domains_no_redundant_gap.tab"
#step A didn't give any result so we used the next steps to run Interproscan locally

#step B.1 Run locally interpro for remaining sequences that not found from the no_redundant_gap
#step B.2 Convert interpro matches to tabular file (main.py)

#step C.1 mapping fasta files to domains found by locally running interpro version 75.0
fasta_dir = "/home/damian/Documents/L3S/projects/datasets/targetp/fasta_set"
domains_path = "/home/damian/Documents/L3S/projects/datasets/targetp/preprocess_overlap/id_domains_overlap_no_gap_total.tab" #step C
output_path = "/home/damian/Documents/L3S/projects/datasets/targetp"
TargetPExperiment = TargetPExperiment(fasta_dir, domains_path, output_path)
TargetPExperiment.fasta2csv()

#step D.1 creating default domains (main.py)
#step D.2 mapping fast files to tabular defaults domains
# fasta_dir = "/home/damian/Documents/L3S/projects/datasets/targetp/remain_seq2"
# domains_path = "/home/damian/Documents/L3S/projects/datasets/targetp/default_domains4remain_seq2/default_dom_remain_seq2.tab"
# output_path = "/home/damian/Documents/L3S/projects/datasets/targetp"
# TargetPExperiment = TargetPExperiment(fasta_dir, domains_path, output_path)
# TargetPExperiment.fasta2csv()
"""

###
# create TargetP data set
###
"""
#=== remove duplicate, shuffle & k-fold split ===#
#place initial dataset.csv to the output folder then the train,test and cross-validation sets will be created.
print("=== ~ ===")
print("2.1) data set -> shuffled data set.")
output_path = "/home/damian/Documents/L3S/projects/datasets/targetp"
label_name = "cellular_location"
dataset_name = "targetp_dataset"
output_path = join(output_path, label_name)
data_set_path = output_path

PrepareDataSet = PrepareDataSet(data_set_path, output_path, dataset_name)
used_columns = ["uniprot_id", "cellular_location", "seq", "interpro_domains"]
PrepareDataSet.read_dataset(used_columns)
# clean from non Interpro domains
PrepareDataSet.remove_no_interpro_domains(used_columns)
# remove duplicate instances
PrepareDataSet.remove_duplicate_instances("interpro_domains", None)
PrepareDataSet.remove_duplicate_instances("seq", None)
# check the number of protein-length unknown domains
PrepareDataSet.count_num_proteins_per_dom_type()
# remove proteins with no interpro domains
PrepareDataSet.remove_unk_gap_proteins()
# shuffle data set
PrepareDataSet.shuffle_dataset()
remove_ids = False
PrepareDataSet.save_dataset(remove_ids)

print("=====")
print("3.1) split train/test stratified on label")
## split train/test stratified on y ##
test_portion = 0.3
PrepareDataSet.split_train_test_stratified(used_columns, label_name, test_portion)

print("=====")
print("3.2) check domains distribution in train and test")
output_path = "/home/damian/Documents/L3S/projects/datasets/targetp"
label_name = "cellular_location"
train_name = "targetp_dataset_train.csv"
test_name = "targetp_dataset_test.csv"
use_test4analysis = True
unk_domains_exist = False
PrepareDataSet.diagnose_oov_domains(train_name, test_name, use_test4analysis, unk_domains_exist)

print("=====")
print("3.3) create inner 3 fold cross validation sets")
## inner fold cross validation stratified on y ##
x_columns = ["seq", "interpro_domains"]
k = 3
PrepareDataSet.split_inner_stratified_kfold(k, x_columns, label_name)

print("=====")
print("3.4) subsample train set")
percentages, num_picks = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5], 20
PrepareDataSet.subset_train(percentages, num_picks)

print("=====")
print("4) split test per OOV")
output_path = "/home/damian/Documents/L3S/projects/datasets/targetp"
label_name = "cellular_location"
dataset_name = "targetp_dataset"
output_path = join(output_path, label_name)
data_set_path = output_path

train_file = "targetp_dataset_train.csv"
test_file = "targetp_dataset_test.csv"
oov_percentages = [0, 0.1, 0.3, 0.5, 0.7, 1.0]
PrepareDataSet.split_test_per_oov_percentage(train_file, test_file, oov_percentages, used_columns)
print("=== ~ ===")
"""

###                  ###
### Toxin Experiment ###
###                  ###
"""
###
# from input fasta to Toxin data with sequence and domains and label columns
###
print("=== Toxin Experiment ===")
print("1) 2 classes fasta -> dataset csv.")
# step A. mapping fasta files to interpro version 75.0
fasta_dir = "/home/damian/Documents/L3S/projects/datasets/toxin/fasta_set"
domains_path = "/home/damian/Documents/L3S/projects/linear/no_red_gap/id_domains_no_redundant_gap.tab"
output_path = "/home/damian/Documents/L3S/projects/datasets/toxin"
ToxinExperiment = ToxinExperiment(fasta_dir, domains_path, output_path)
is_local_run = False
ToxinExperiment.fasta2csv()

#step B.1 Run locally interpro for remaining sequences that not found from the no_redundant_gap
#step B.2 Convert interpro matches to tabular file (main.py)

#step C.1 mapping fasta files to domains found by locallily running interpro version 75.0
fasta_dir = "/home/damian/Documents/L3S/projects/datasets/toxin/remain_seqs1"
domains_path = "/home/damian/Documents/L3S/projects/datasets/toxin/id_domains_no_redundant_gap_remain1_all.tab"
output_path = "/home/damian/Documents/L3S/projects/datasets/toxin"
ToxinExperiment = ToxinExperiment(fasta_dir, domains_path, output_path)
is_local_run = True
ToxinExperiment.fasta2csv(is_local_run)

#step D.1 creating default domains (main.py)
#step D.2 mapping fast files to tabular defaults domains
fasta_dir = "/home/damian/Documents/L3S/projects/datasets/toxin/remain_seq2"
domains_path = "/home/damian/Documents/L3S/projects/datasets/toxin/default_domains_remain2_all.tab"
output_path = "/home/damian/Documents/L3S/projects/datasets/toxin"
ToxinExperiment = ToxinExperiment(fasta_dir, domains_path, output_path)
is_local_run = True
ToxinExperiment.fasta2csv(is_local_run)
"""

###
# create Toxin data set splits
###
"""
# === remove duplicate, shuffle & k-fold split ===#
# place initial dataset.csv to the output folder then the train,test and cross-validation sets will be created.
print("2) data set -> clean from non Interpro -> remove duplicates -> shuffled data set.")
output_path = "/home/damian/Documents/L3S/projects/datasets/toxin"
label_name = "toxin"
dataset_name = "toxin_dataset"
output_path = join(output_path, label_name)
data_set_path = output_path

PrepareDataSet = PrepareDataSet(data_set_path, output_path, dataset_name)
used_columns = ["uniprot_id", "toxin", "seq", "interpro_domains"]
PrepareDataSet.read_dataset(used_columns)
# clean from non Interpro domains
PrepareDataSet.remove_no_interpro_domains(used_columns)
# remove duplicate instances
PrepareDataSet.remove_duplicate_instances("interpro_domains", label_name)
# check the number of protein-length unknown domains
PrepareDataSet.count_num_proteins_per_dom_type()
# remove proteins with no interpro domains
PrepareDataSet.remove_unk_gap_proteins()
# shuffle data set
PrepareDataSet.shuffle_dataset()
remove_ids = False
PrepareDataSet.save_dataset(remove_ids)

print("=====")
print("3.1) split train/test stratified on label")
used_columns_split = ["toxin", "seq", "interpro_domains"]
test_portion = 0.3
PrepareDataSet.split_train_test_stratified(used_columns, label_name, test_portion)

print("=====")
print("3.2) Check domains distribution in train and test")
output_path = "/home/damian/Documents/L3S/projects/datasets/toxin"
label_name = "toxin"
data_set_path = join(output_path, label_name)
output_path = join(output_path, label_name)
dataset_name = "toxin_" + "dataset"
PrepareDataSet = PrepareDataSet(data_set_path, output_path, dataset_name)
train_name = "toxin_dataset_train.csv"
test_name = "toxin_dataset_test.csv"
use_test4analysis = True
unk_domains_exist = False
PrepareDataSet.diagnose_oov_domains(train_name, test_name, use_test4analysis, unk_domains_exist)

print("=====")
print("3.3) create inner 3 fold cross validation sets")
x_columns = ["seq", "interpro_domains"]
k = 3
PrepareDataSet.split_inner_stratified_kfold(k, x_columns, label_name)

print("=====")
print("3.4) subsample train set")
percentages, num_picks = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5], 20
PrepareDataSet.subset_train(percentages, num_picks)
"""

###                ###
### NEW Experiment ###
###                ###
"""
###
# from input fasta to NEW data with sequence and domains and label columns
###
print("=== NEW Experiment ===")
# print("1) txt -> fasta.")
input_path = "/home/damian/Documents/L3S/projects/datasets/new"
domains_path = "/home/damian/Documents/L3S/projects/datasets/new/id_domains_no_redundant_gap.tab"
output_path = "/home/damian/Documents/L3S/projects/datasets/new"
# NEWExperiment = NEWExperiment(input_path, domains_path, output_path)
#NEWExperiment.txt2fastas("new_data_label_sequence.txt")

print("2) fasta -> csv.")
print("a) Run local interpro and create tabular file")
#Step A. Data set does not contain uniprot ids,
#1. so run locally interpro for starting fasta file
#2. create tabular file for domains (main.py)

print("b) save proteins with found domains to data set csv.")
#Step B. mapping fasta files to found domains
# NEWExperiment = NEWExperiment(input_path, domains_path, output_path)
# fasta_name = "new_dataset.fasta"
# NEWExperiment.fasta2csv(fasta_name)

print("c) create default domains and append the respective proteins to the csv.")
#Step C. create default domains for remaining proteins add them to the data set csv
#1. creating default domains for remaining sequences (main.py)
#2. mapping remaining fasta to default domains tabular file
remain_fasta_name = "new_remaining_seq1.fasta"
domains_path = "/home/damian/Documents/L3S/projects/datasets/new/default_domains.tab"
NEWExperiment = NEWExperiment(input_path, domains_path, output_path)
NEWExperiment.fasta2csv(remain_fasta_name)
"""

###
# create NEW data set splits
###
"""
#=== remove duplicate, shuffle & k-fold split ===#
#place initial dataset.csv to the output folder then the train,test and cross-validation sets will be created.
print("2) data set -> shuffled data set.")
output_path = "/home/damian/Documents/L3S/projects/datasets/new"
label_name = "ec"
dataset_name = "new_dataset"
output_path = join(output_path, label_name)
data_set_path = output_path  # when you have one csv file, leave it on the data set out and it will create the fold on the output folder
PrepareDataSet = PrepareDataSet(data_set_path, output_path, dataset_name)
used_columns = ["id", "ec", "seq", "interpro_domains"]
PrepareDataSet.read_dataset(used_columns)
# clean from non Interpro domains
PrepareDataSet.remove_no_interpro_domains(used_columns)
# remove duplicate instances
PrepareDataSet.remove_duplicate_instances("interpro_domains", label_name)
# check the number of protein-length unknown domains
PrepareDataSet.count_num_proteins_per_dom_type()
# remove proteins with no interpro domains
PrepareDataSet.remove_unk_gap_proteins()
#shuffle data set
PrepareDataSet.shuffle_dataset()
remove_ids = False
PrepareDataSet.save_dataset(remove_ids)

print("=====")
print("3.1) split train/test stratified on label")
test_portion = 0.3
PrepareDataSet.split_train_test_stratified(used_columns,label_name,test_portion)

print("=====")
print("3.2) Check domains distribution in train and test")
output_path = "/home/damian/Documents/L3S/projects/datasets/new"
label_name = "ec"
data_set_path = join(output_path, label_name)
output_path = join(output_path, label_name)
dataset_name = "new_dataset"
# PrepareDataSet = PrepareDataSet(data_set_path, output_path, dataset_name)
train_name = "new_dataset_train.csv"
test_name = "new_dataset_test.csv"
use_test4analysis = True
unk_domains_exist = False
PrepareDataSet.diagnose_oov_domains(train_name, test_name, use_test4analysis, unk_domains_exist)

print("=====")
print("3.3) create inner 3-fold cross validation sets")
x_columns = ["seq", "interpro_domains"]
k = 3
PrepareDataSet.split_inner_stratified_kfold(k,x_columns,label_name)

print("=====")
print("3.4) subsample train set")
percentages, num_picks = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5], 20
PrepareDataSet.subset_train(percentages, num_picks)
print("=== ===")
"""
