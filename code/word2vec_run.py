from timeit import default_timer as timer
from WordEmb import WordEmb
from utils import sec2hour_min_sec
import argparse

"""
Set up and run word2vec embeddings
"""
parser = argparse.ArgumentParser(description="Set up and run word2vec.")
#File path
parser.add_argument('--data_path',help="Data path for input and output")
parser.add_argument('--file_in',help="Corpus file name")
#Word2Vec
parser.add_argument('--window', type=int, help="Window size")
parser.add_argument('--use_skipgram', type=int, help="CBOW:0, Skip-gram:1")
parser.add_argument('--vec_dim', type=int, help="Vectors dimension")
parser.add_argument('--cores',type=int,help="Number of cores")
parser.add_argument('--max_epoches',type=int,help="Maximum number of epoches to train")
parser.add_argument('--epoches_step',type=int,help="Epoch step size")
args = parser.parse_args()
# Server test
data_path = args.data_path #"/data2/melidis"  # "/home/damian/Documents/L3S/projects"#

print("\n=====")
print("4) Train domains_copurs.txt -> dom2vec.txt")
file_in = args.file_in #"domains_corpus_no_redundant_gap.txt"#"domains_corpus_no_overlap.txt"#"domains_corpus_prep1.txt"

#Train step-wise
### Word2vec Parameters ###
window = args.window
use_skipgram = args.use_skipgram
assert use_skipgram == 0 or use_skipgram == 1, "AssertionError: use_skipgram can be only 0(CBOW) or 1(Skipgram)"
use_hierachical_soft_max = 0
vec_dim = args.vec_dim
cores = args.cores
epochs_step = args.epoches_step
max_epochs = args.max_epoches
### ###

time_start = timer()
dom2Vec = WordEmb(data_path, file_in)
dom2Vec.set_up(window, use_skipgram, use_hierachical_soft_max, vec_dim,cores)
dom2Vec.build_voc()
dom2Vec.train_stepwise(max_epochs,epochs_step)
time_end=timer()
print("Elapsed CPU time for initializing and training the model: {}.".format(sec2hour_min_sec(time_end-time_start)))
