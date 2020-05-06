import os
from ParentChildEvaluate import ParentChildEvaluate
from PlotEmbTypes import PlotEmbTypes
from ParseInterPro2GO import ParseInterPro2GO
from GOEvaluate import GOEvaluate
from GOSimEvaluate import GOSimEvaluate
from DomainXMLParser import DomainXMLParser
from EC_SCOP_Evaluate import EC_SCOP_Evaluate
from utils import write_random_vectors, create_dir
import itertools
import matplotlib.pyplot as plt
import numpy as np

print("=== Intrinsic Evaluation ===")
###                       ###
### Random domain vectors ###
###                       ###
"""
print("=== Create random domain vectors to test significance of results ===")
window = [2]
is_skipgram = [0] #[0,1]
emb_dim = [50, 100, 200] #[50,100,200]
epochs = [5]
all_emb_instances = list(itertools.product(window, is_skipgram,emb_dim, epochs))
### Embedding type###
### Linear ###
# model_path = "/home/damian/Documents/L3S/projects/linear_gap/no_red_gap"  # no redundant domains with gap
model_path = "/home/damian/Documents/L3S/projects/linear_gap/no_overlap" #no overlap
rand_model_path = os.path.join(model_path, "random")
create_dir(rand_model_path)

for emb_instance in all_emb_instances:
	model_file_name = "dom2vec" + "_w" + str(emb_instance[0]) + "_sg" + str(
		emb_instance[1]) + "_hierSoft0" + "_dim" + str(emb_instance[2]) + "_e" + str(emb_instance[3]) + ".txt"
	model_file = os.path.join(model_path, model_file_name)
	write_random_vectors(rand_model_path,model_file)

print("=== ===")
"""

###             ###
### ParentChild ###
###             ###
"""
print("1. Evaluate embeddings: Precision for ParentChild relation")
# create all trained embeddings possible instances
###dom2vec###
window = [2]  # [2, 5]
is_skipgram = [1]  # [0, 1] #0: CBOW, 1: Skipgram
emb_dim = [200]  # [50, 100, 200]
epochs = [5]  # [e for e in range(5, 55, 5)]

###random vec###
#window = [2]
#is_skipgram = [0]
#emb_dim = [50, 100, 200]
#epochs = [5]

all_emb_instances = list(itertools.product(window, is_skipgram, emb_dim, epochs))
### Embedding type###
### Linear ###
model_path = "/home/damian/Documents/L3S/projects/linear_gap/no_red_gap"  # no redundant domains with gap
# model_path = "/home/damian/Documents/L3S/projects/linear_gap/no_overlap" #no overlap with gap

load_rand_vec = False #True  # load dom2vec or random vectors
if load_rand_vec:
	model_path = os.path.join(model_path, "random")
data_path = model_path
parent_child_eval = ParentChildEvaluate(data_path)

# create the tree
interpro_tree_path = "/home/damian/Desktop/domains/interpro_tree"
interpro_tree_file_name = "ParentChildTreeFile.txt"
interpro_tree_file = os.path.join(interpro_tree_path, interpro_tree_file_name)
parsed_tree_file_name = "interpro_parsed_tree.txt"  # Output file
save_parsed_tree = False
parent_child_eval.parse_parent_child_file(interpro_tree_file, interpro_tree_path, parsed_tree_file_name, save_parsed_tree)

is_model_bin = False
for emb_instance in all_emb_instances:
	print(emb_instance)
	# set up the embedding file
	if load_rand_vec:
		print("Loading random vectors")
		model_file_name = "dom2vec" + "_w2" + "_sg" + str(emb_instance[1]) + "_hierSoft0" + "_dim" + str(
			emb_instance[2]) + "_e5" + "_rand" + ".txt"
	else:
		model_file_name = "dom2vec" + "_w" + str(emb_instance[0]) + "_sg" + str(
			emb_instance[1]) + "_hierSoft0" + "_dim" + str(emb_instance[2]) + "_e" + str(emb_instance[3]) + ".txt"
	model_file = os.path.join(model_path, model_file_name)

	parent_child_eval.load_emb_model(model_file, is_model_bin)
	plot_histograms = True
	save_diagnostics = False
	parent_child_eval.get_nn_calculate_precision_recall_atN(100, plot_histograms, save_diagnostics)
	print("---")
print("--- ---")
"""

###                 ###
###    GOEvaluate   ###
###                 ###

###                                          ###
### Preprocess interpro2go.txt for GO        ###
###                                          ###
"""
print("interpro2GO -> interpro2GO tabs")
interpro2go = "/home/damian/Desktop/domains/go/interpro2go.txt"
# data_path = "/home/damian/Desktop/domains/go/human"
# species_domains = "interpro_dom_human.txt"
# species_name = "human"
# data_path = "/home/damian/Desktop/domains/go/ecolik12"
# species_domains = "interpro_dom_ecolik12.txt"
# species_name = "ecolik12"
data_path = "/home/damian/Desktop/domains/go/yeast"
species_domains = "interpro_dom_yeast.txt"
species_name = "yeast"
# data_path = "/home/damian/Desktop/domains/go/malaria"
# species_domains = "interpro_dom_malaria.txt"
# species_name = "malaria"
prepareInterPro2GO = ParseInterPro2GO(data_path, interpro2go, species_domains, species_name)
# prepareInterPro2GO.run(keep_only_MF=True, num_comb=50000) #place the species domains file in the data_path and let it run..
prepareInterPro2GO.convert_go_labels(keep_only_MF=True)#get first level GO annotations as labels
"""

###                                                ###
### Evaluate embeddings with GO molecular function ###
###                                                ###

# print("Predict 1-level GO annotation using k-NN -> evaluate embeddings")
# create all trained embeddings possible instances
### dom2vec ###
# window = [2] #[2, 5]#[2, 5]
# is_skipgram = [0] #[0, 1] #[0, 1] #0: CBOW, 1: Skipgram
# emb_dim = [50] #[50, 100, 200]
# epochs = [5] #[e for e in range(5, 55, 5)]

### random vec ###
# window = [2]
# is_skipgram = [0]
# emb_dim = [50, 100, 200]
# epochs = [5]
#
# all_emb_instances = list(itertools.product(window, is_skipgram, emb_dim, epochs))
# is_model_bin = False

###       ###
### human ###
###       ###
"""
print("=== human ===")
data_path_human = "/home/damian/Desktop/domains/go/human"
domains_go_file = "interpro2go_human_MF_labels.csv"
go_evaluate_human = GOEvaluate(data_path_human, domains_go_file)
print("=== No overlap ===")
emb_path = "/home/damian/Documents/L3S/projects/linear_gap/no_overlap"
load_rand_vec = True  # load dom2vec or random vectors
if load_rand_vec:
	emb_path = os.path.join(emb_path, "random")

use_shortest = False  # use label of parent with shortest level

for emb_instance in all_emb_instances:
	print(emb_instance)
	# set up the embedding file
	if load_rand_vec:
		print("Loading random vectors")
		model_file_name = "dom2vec" + "_w" + str(emb_instance[0]) + "_sg" + str(emb_instance[1]) + "_hierSoft0" + "_dim" \
		                  + str(emb_instance[2]) + "_e" + str(emb_instance[3]) + "_rand" + ".txt"
	else:
		model_file_name = "dom2vec" + "_w" + str(emb_instance[0]) + "_sg" + str(emb_instance[1]) + "_hierSoft0" + "_dim" \
		                  + str(emb_instance[2]) + "_e" + str(emb_instance[3]) + ".txt"
	emb_file = os.path.join(emb_path, model_file_name)
	go_evaluate_human.run_classification(use_shortest, emb_file, is_model_bin)
	print("---")


print("=== No redundant ===")
emb_path = "/home/damian/Documents/L3S/projects/linear_gap/no_red_gap"
if load_rand_vec:
	emb_path = os.path.join(emb_path, "random")

# True: use label of parent with lowest level (= 1) immediate child of root,
# False: use label of all parents with lowest level (=1)
use_shortest = False

for emb_instance in all_emb_instances:
	print(emb_instance)
	# set up the embedding file
	if load_rand_vec:
		print("Loading random vectors")
		model_file_name = "dom2vec" + "_w" + str(emb_instance[0]) + "_sg" + str(emb_instance[1]) + "_hierSoft0" + "_dim" \
	                  + str(emb_instance[2]) + "_e" + str(emb_instance[3]) + "_rand" + ".txt"
	else:
		model_file_name = "dom2vec" + "_w" + str(emb_instance[0]) + "_sg" + str(emb_instance[1]) + "_hierSoft0" + "_dim" \
	                  + str(emb_instance[2]) + "_e" + str(emb_instance[3]) + ".txt"
	emb_file = os.path.join(emb_path, model_file_name)
	go_evaluate_human.run_classification(use_shortest, emb_file, is_model_bin)
	print("---")
print("=== ===")
"""

###       ###
### yeast ###
###       ###
"""
print("=== yeast ===")
data_path_yeast = "/home/damian/Desktop/domains/go/yeast"
domains_go_file = "interpro2go_yeast_MF_labels.csv"
go_evaluate_yeast = GOEvaluate(data_path_yeast, domains_go_file)

print("=== No overlap ===")
emb_path = "/home/damian/Documents/L3S/projects/linear_gap/no_overlap"
load_rand_vec = True  # load dom2vec or random vectors
if load_rand_vec:
	emb_path = os.path.join(emb_path, "random")

use_shortest = False  # use label of parent with shortest level
for emb_instance in all_emb_instances:
	print(emb_instance)
	# set up the embedding file
	if load_rand_vec:
		print("Loading random vectors")
		model_file_name = "dom2vec" + "_w" + str(emb_instance[0]) + "_sg" + str(emb_instance[1]) + "_hierSoft0" + "_dim" \
		                  + str(emb_instance[2]) + "_e" + str(emb_instance[3]) +"_rand" + ".txt"
	else:
		model_file_name = "dom2vec" + "_w" + str(emb_instance[0]) + "_sg" + str(emb_instance[1]) + "_hierSoft0" + "_dim" \
	                  + str(emb_instance[2]) + "_e" + str(emb_instance[3]) + ".txt"
	emb_file = os.path.join(emb_path, model_file_name)
	go_evaluate_yeast.run_classification(use_shortest, emb_file, is_model_bin)

print("=== No redundant ===")
emb_path = "/home/damian/Documents/L3S/projects/linear_gap/no_red_gap"
if load_rand_vec:
	emb_path = os.path.join(emb_path, "random")

use_shortest = False  # use label of parent with shortest level

for emb_instance in all_emb_instances:
	print(emb_instance)
	# set up the embedding file
	if load_rand_vec:
		print("Loading random vectors")
		model_file_name = "dom2vec" + "_w" + str(emb_instance[0]) + "_sg" + str(emb_instance[1]) + "_hierSoft0" + "_dim" \
		+ str(emb_instance[2]) + "_e" + str(emb_instance[3]) + "_rand" + ".txt"
	else:
		model_file_name = "dom2vec" + "_w" + str(emb_instance[0]) + "_sg" + str(emb_instance[1]) + "_hierSoft0" + "_dim" \
	                  + str(emb_instance[2]) + "_e" + str(emb_instance[3]) + ".txt"
	emb_file = os.path.join(emb_path, model_file_name)
	go_evaluate_yeast.run_classification(use_shortest, emb_file, is_model_bin)
print("=== ===")
"""

###          ###
### ecolik12 ###
###          ###
"""
print("=== ecolik12 ===")
data_path_ecoli = "/home/damian/Desktop/domains/go/ecolik12"
domains_go_file = "interpro2go_ecolik12_MF_labels.csv"
go_evaluate_ecoli = GOEvaluate(data_path_ecoli, domains_go_file)
print("=== No overlap ===")
emb_path = "/home/damian/Documents/L3S/projects/linear_gap/no_overlap"
load_rand_vec = True  # load dom2vec or random vectors
if load_rand_vec:
	emb_path = os.path.join(emb_path, "random")

use_shortest = False  # use label of parent with shortest level

for emb_instance in all_emb_instances:
	print(emb_instance)
	# set up the embedding file
	if load_rand_vec:
		print("Loading random vectors")
		model_file_name = "dom2vec" + "_w" + str(emb_instance[0]) + "_sg" + str(emb_instance[1]) + "_hierSoft0" + "_dim" \
		                  + str(emb_instance[2]) + "_e" + str(emb_instance[3]) + "_rand" + ".txt"
	else:
		model_file_name = "dom2vec" + "_w" + str(emb_instance[0]) + "_sg" + str(emb_instance[1]) + "_hierSoft0" + "_dim" \
		                  + str(emb_instance[2]) + "_e" + str(emb_instance[3]) + ".txt"
	emb_file = os.path.join(emb_path, model_file_name)
	go_evaluate_ecoli.run_classification(use_shortest, emb_file, is_model_bin)

print("=== No redundant ===")
emb_path = "/home/damian/Documents/L3S/projects/linear_gap/no_red_gap"
if load_rand_vec:
	emb_path = os.path.join(emb_path, "random")
use_shortest = False  # use label of parent with shortest level

for emb_instance in all_emb_instances:
	print(emb_instance)
	# set up the embedding file
	if load_rand_vec:
		print("Loading random vectors")
		model_file_name = "dom2vec" + "_w" + str(emb_instance[0]) + "_sg" + str(emb_instance[1]) + "_hierSoft0" + "_dim" \
		                  + str(emb_instance[2]) + "_e" + str(emb_instance[3]) + "_rand" + ".txt"
	else:
		model_file_name = "dom2vec" + "_w" + str(emb_instance[0]) + "_sg" + str(emb_instance[1]) + "_hierSoft0" + "_dim" \
		                  + str(emb_instance[2]) + "_e" + str(emb_instance[3]) + ".txt"
	emb_file = os.path.join(emb_path, model_file_name)
	go_evaluate_ecoli.run_classification(use_shortest, emb_file, is_model_bin)
print("=== ===")
"""

###         ###
### malaria ###
###         ###
"""
print("=== malaria ===")
data_path_malaria = "/home/damian/Desktop/domains/go/malaria"
domains_go_file = "interpro2go_malaria_MF_labels.csv"
go_evaluate_malaria = GOEvaluate(data_path_malaria, domains_go_file)

print("=== No overlap ===")
emb_path = "/home/damian/Documents/L3S/projects/linear_gap/no_overlap"
load_rand_vec = True  # load dom2vec or random vectors
if load_rand_vec:
	emb_path = os.path.join(emb_path, "random")

use_shortest = False  # use label of parent with shortest level
for emb_instance in all_emb_instances:
	print(emb_instance)
	# set up the embedding file
	if load_rand_vec:
		print("Loading random vectors")
		model_file_name = "dom2vec" + "_w" + str(emb_instance[0]) + "_sg" + str(emb_instance[1]) + "_hierSoft0" + "_dim" \
		                  + str(emb_instance[2]) + "_e" + str(emb_instance[3]) + "_rand" + ".txt"
	else:
		model_file_name = "dom2vec" + "_w" + str(emb_instance[0]) + "_sg" + str(emb_instance[1]) + "_hierSoft0" + "_dim" \
		                  + str(emb_instance[2]) + "_e" + str(emb_instance[3]) + ".txt"
	emb_file = os.path.join(emb_path, model_file_name)
	go_evaluate_malaria.run_classification(use_shortest, emb_file, is_model_bin)

print("=== No redundant ===")
emb_path = "/home/damian/Documents/L3S/projects/linear_gap/no_red_gap"
if load_rand_vec:
	emb_path = os.path.join(emb_path, "random")

use_shortest = False  # use label of parent with shortest level

for emb_instance in all_emb_instances:
	print(emb_instance)
	# set up the embedding file
	if load_rand_vec:
		print("Load random vectors")
		model_file_name = "dom2vec" + "_w" + str(emb_instance[0]) + "_sg" + str(emb_instance[1]) + "_hierSoft0" + "_dim" \
		                  + str(emb_instance[2]) + "_e" + str(emb_instance[3]) + "_rand" + ".txt"
	else:
		model_file_name = "dom2vec" + "_w" + str(emb_instance[0]) + "_sg" + str(emb_instance[1]) + "_hierSoft0" + "_dim" \
		                  + str(emb_instance[2]) + "_e" + str(emb_instance[3]) + ".txt"
	emb_file = os.path.join(emb_path, model_file_name)
	go_evaluate_malaria.run_classification(use_shortest, emb_file, is_model_bin)
print("=== ===")
"""

###             ###
### EC & SCOPe  ###
###             ###

###                                          ###
### Preprocess interpro.xml for EC and SCOPe ###
###                                          ###
"""
#get association id -> EC and SCOPe
print("EC and SCOPe -> k-NN Performance")

### DomainXMLParser - Parse EC and SCOPe for interpro ###
print("1. Get interpro_id -> EC and SCOPe")
data_path = "/home/damian/Desktop/domains"
interpro_xml = "interpro.xml"
out_name = "interpro2EC_SCOPe.tab"
domainXMLParser = DomainXMLParser(data_path, interpro_xml, out_name)
domainXMLParser.parse_and_save_EC_SCOP()
"""

###                                      ###
### Evaluate embeddings using EC & SCOPe ###
###                                      ###
"""
data_path = "/home/damian/Desktop/domains"
dom_ec_scope_file = "interpro2EC_SCOPe.tab"
### Embedding type###
# create all trained embeddings possible instances
###dom2vec###
# window = [2,5] #[2, 5]
# is_skipgram = [0,1] #[0, 1] #0: CBOW, 1: Skipgram
# emb_dim = [50, 100, 200]
# epochs = [e for e in range(5, 55, 5)]

###random vec###
window = [2]
is_skipgram = [0]
emb_dim = [50, 100, 200]
epochs = [5]
all_emb_instances = list(itertools.product(window, is_skipgram, emb_dim, epochs))
"""

###         ###
###  SCOPe  ###
###         ###
"""
print("3. domain embeddings labeled by SCOPe -> k-NN")
### Linear ###
print("=== no overlap ===")
model_path = "/home/damian/Documents/L3S/projects/linear_gap/no_overlap"
load_rand_vec = True  # load dom2vec or random vectors
if load_rand_vec:
	model_path = os.path.join(model_path, "random")

for emb_instance in all_emb_instances:
	print(emb_instance)
	out_name = "scop_clusters.png"
	use_ec = False
	SCOP_ClusterEvaluate = EC_SCOP_Evaluate(data_path, dom_ec_scope_file, use_ec, out_name)
	is_model_bin = False
	# set up the embedding file
	if load_rand_vec:
		print("Loading random vectors")
		model_file_name = "dom2vec" + "_w" + str(emb_instance[0]) + "_sg" + str(emb_instance[1]) + "_hierSoft0" + "_dim" \
		                  + str(emb_instance[2]) + "_e" + str(emb_instance[3]) + "_rand" + ".txt"
	else:
		model_file_name = "dom2vec" + "_w" + str(emb_instance[0]) + "_sg" + str(emb_instance[1]) + "_hierSoft0" + "_dim" \
		                  + str(emb_instance[2]) + "_e" + str(emb_instance[3]) + ".txt"
	model_file = os.path.join(model_path, model_file_name)

	SCOP_ClusterEvaluate.run_classification(model_file, is_model_bin, dim_reduction_algo="", low_dim_size=10,
	                         classifier_name="NN")

	print("---")

print("=== no red gap ===")
model_path = "/home/damian/Documents/L3S/projects/linear_gap/no_red_gap"
load_rand_vec = True  # load dom2vec or random vectors
if load_rand_vec:
	model_path = os.path.join(model_path, "random")

for emb_instance in all_emb_instances:
	print(emb_instance)
	out_name = "scop_clusters.png"
	use_ec = False
	SCOP_ClusterEvaluate = EC_SCOP_Evaluate(data_path, dom_ec_scope_file, use_ec, out_name)
	is_model_bin = False
	# set up the embedding file
	if load_rand_vec:
		print("Loading random vectors")
		model_file_name = "dom2vec" + "_w" + str(emb_instance[0]) + "_sg" + str(emb_instance[1]) + "_hierSoft0" + "_dim" \
		                  + str(emb_instance[2]) + "_e" + str(emb_instance[3]) + "_rand" + ".txt"
	else:
		model_file_name = "dom2vec" + "_w" + str(emb_instance[0]) + "_sg" + str(emb_instance[1]) + "_hierSoft0" + "_dim" \
		                  + str(emb_instance[2]) + "_e" + str(emb_instance[3]) + ".txt"
	model_file = os.path.join(model_path, model_file_name)

	SCOP_ClusterEvaluate.run_classification(model_file, is_model_bin, dim_reduction_algo="", low_dim_size=10,
	                         classifier_name="NN")  # run lle, pca,t-sne

	print("---")
"""

###    ###
### EC ###
###    ###
"""
print("2. domain embeddings labeled by EC -> k-NN")
print("=== no overlap ===")
### Embedding type###
### Linear ###
model_path = "/home/damian/Documents/L3S/projects/linear_gap/no_overlap"
load_rand_vec = True  # load dom2vec or random vectors
if load_rand_vec:
	model_path = os.path.join(model_path, "random")

for emb_instance in all_emb_instances:
	print(emb_instance)
	out_name = "scop_clusters.png"
	use_ec = True
	EC_ClusterEvaluate_no_overlap = EC_SCOP_Evaluate(data_path, dom_ec_scope_file, use_ec, out_name)
	is_model_bin = False
	#set up the embedding file
	if load_rand_vec:
		print("Loading random vectors")
		model_file_name = "dom2vec" + "_w" + str(emb_instance[0]) + "_sg" + str(emb_instance[1]) + "_hierSoft0" + "_dim" \
		                  + str(emb_instance[2]) + "_e" + str(emb_instance[3]) + "_rand" + ".txt"
	else:
		model_file_name = "dom2vec" + "_w" + str(emb_instance[0]) + "_sg" + str(emb_instance[1]) + "_hierSoft0"+"_dim" \
	                  + str(emb_instance[2])+"_e" + str(emb_instance[3]) + ".txt"
	model_file = os.path.join(model_path, model_file_name)
	mode_file = os.path.join(model_path, model_file_name)
	EC_ClusterEvaluate_no_overlap.run_classification(model_file, is_model_bin, dim_reduction_algo="", low_dim_size=10,
	                         classifier_name="NN")  # run lle, pca,t-sne
	print("---")

print("=== no redundant ===")
### Embedding type###
### Linear ###
model_path = "/home/damian/Documents/L3S/projects/linear_gap/no_red_gap"
load_rand_vec = True  # load dom2vec or random vectors
if load_rand_vec:
	model_path = os.path.join(model_path, "random")

for emb_instance in all_emb_instances:
	print(emb_instance)
	out_name = "ec_clusters.png"
	use_ec = True
	EC_ClusterEvaluate_no_red_gap = EC_SCOP_Evaluate(data_path, dom_ec_scope_file, use_ec, out_name)
	is_model_bin = False
	#set up the embedding file
	if load_rand_vec:
		print("Loading random vectors")
		model_file_name = "dom2vec" + "_w" + str(emb_instance[0]) + "_sg" + str(emb_instance[1]) + "_hierSoft0" + "_dim" \
		                  + str(emb_instance[2]) + "_e" + str(emb_instance[3]) + "_rand" + ".txt"
	else:
		model_file_name = "dom2vec" + "_w" + str(emb_instance[0]) + "_sg" + str(emb_instance[1]) + "_hierSoft0"+"_dim" \
	                  + str(emb_instance[2])+"_e" + str(emb_instance[3]) + ".txt"
	model_file = os.path.join(model_path, model_file_name)
	mode_file = os.path.join(model_path, model_file_name)
	EC_ClusterEvaluate_no_red_gap.run_classification(model_file, is_model_bin, dim_reduction_algo="", low_dim_size=10,
	                         classifier_name="NN")  # run lle, pca,t-sne
	print("---")
"""
print("=== * ===")
print("== *** ==")
