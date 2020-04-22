from SetUpNeurNet import SetUpNeurNet
from NeurNet import NeurNet
from os.path import join
import argparse

"""
Set up and run neural net
"""
parser = argparse.ArgumentParser(description="Set up and run neural net.")
# Data set
parser.add_argument('--dataset_name', help="Data set name")
parser.add_argument('--data_path', help="Data path")
parser.add_argument('--label_name', help="Label name")
parser.add_argument('--train_file', help="Data train file")
parser.add_argument('--test_file', help="Data test file")
# Embeddings
parser.add_argument('--emb_name', help="protvec, seqvec or dom2vec")
parser.add_argument('--use_emb', type=int, help="Use or not embeddings")
parser.add_argument('--emb_file', help="Embeddings file")
parser.add_argument('--emb_bin', type=int, help="Is embedding bin or not")
parser.add_argument('--freeze_emb', type=int, help="Freeze embeddings: 0/1")
parser.add_argument('--normalize_emb', type=int, help="Normalize embeddings: 0/1")
# Model General
parser.add_argument('--k_fold', type=int, help="k-fold")
parser.add_argument('--model_type', help="Model type")
parser.add_argument('--batch_size', type=int, help="Batch size")
parser.add_argument("--learning_rate", type=float, help="Learning rate of optimizer")
parser.add_argument("--epoches", type=int, help="Number of epoches")
parser.add_argument("--weight_decay", type=float, help="Weight decay")
# CNN
parser.add_argument('--num_filters', type=int, help="Number of filters", required=False)
parser.add_argument('--filter_sizes', help="Filter sizes", required=False)
# LSTM
parser.add_argument('--hid_dim', type=int, help="Hidden dimension of cell", required=False)
parser.add_argument('--is_bidirectional', type=int, help="Bidirectional or not", required=False)
parser.add_argument('--num_layers', type=int, help="Number of layers", required=False)
# FastText
parser.add_argument("--use_uni_bi", type=int, help="Use uni- and bi-grams", required=False)
# both CNN, FastText and LSTM
parser.add_argument('--dropout', type=float, help="Dropout probability", required=False)
parser.add_argument('--save_predictions', type=int, help="Save predictions", required=False)
args = parser.parse_args()

# --- Data set ---#
### DeepLoc ###
dataset_name = args.dataset_name
label_name = args.label_name
data_path = args.data_path
train_file = args.train_file
test_file = args.test_file
### --- ###

# --- Embeddings ---#
emb_name = args.emb_name
use_emb = True if args.use_emb == 1 else False
is_emb_bin = True if args.emb_bin == 1 else False
freeze_emb = True if args.freeze_emb == 1 else False
normalize_emb = True if args.normalize_emb == 1 else False

if dataset_name == "DeepLoc":
	if label_name == "membrane_soluble":
		is_binary_class = True  # membrane_or soluble
	else:
		is_binary_class = False  # location
elif dataset_name == "TargetP":
	is_binary_class = False  # location has 4 possible classes
elif dataset_name == "Toxin":
	is_binary_class = True  # toxin or not toxin
elif dataset_name == "NEW":
	is_binary_class = False  # EC has 6 primary classes

k_fold = args.k_fold
### --- ###

# --- Common ---#
model_type = args.model_type
use_emb = True if args.use_emb == 1 else False
# set up embedding dimensions based on embeddings type
# dom2vec = 50 or 100 or 200, protvec = 100, seqvec = 1024
if emb_name == "protvec":
	if use_emb:
		emb_dim = 100
	else:
		emb_dim = 100
elif emb_name == "dom2vec":
	if use_emb:
		emb_dim = int(args.emb_file.split("_dim")[1].split("_")[0])
	else:
		emb_dim = int(args.emb_file)
	assert emb_dim == 50 or emb_dim == 100 or emb_dim == 200, "AssertionError: dom2vec is trained only for 50, 100, 200"
elif emb_name == "seqvec":
	emb_dim = 1024

dropout = args.dropout
batch_size = args.batch_size
learning_rate = args.learning_rate
weight_decay = args.weight_decay
num_epoches = args.epoches
save_predictions = True if args.save_predictions == 1 else False

# --- Special ---#
print(model_type)
assert model_type == "CNN" or model_type == "LSTM" or model_type == "FastText" or model_type == "SeqVecNet" or model_type == "SeqVecCharNet", "AssertionError: Unknown model type."
use_uni_bi = False #pre-set use_uni_bi for all models to False
if model_type == "CNN":
	# --- CNN ---#
	num_filters = args.num_filters
	filter_sizes = [int(filter_size) for filter_size in args.filter_sizes.split("_")]
	model_dir = model_type + "_num" + str(num_filters) + "_sizes_" + "_".join([str(size) for size in filter_sizes])
elif model_type == "LSTM":
	# --- LSTM ---#
	hid_dim = args.hid_dim
	is_bidirectional = True if args.is_bidirectional == 1 else False
	n_layers = args.num_layers
	model_dir = model_type + "_hid" + str(hid_dim) + "_bi" + str(int(is_bidirectional)) + "_layers" + str(n_layers)
elif model_type == "FastText":
	# --- FastText ---#
	model_dir = model_type
	use_uni_bi = True if args.use_uni_bi == 1 else False
elif model_type == "SeqVecNet":
	# --- SeqVec ---#
	hid_dim = args.hid_dim
	model_dir = model_type + "_hid" + str(hid_dim)
elif model_type == "SeqVecCharNet":
	# --- SeqVecCharNet ---#
	hid_dim = args.hid_dim
	model_dir = model_type + "_hid" + str(hid_dim)

out_path = join(data_path, model_dir)
model_setup = SetUpNeurNet(data_path, train_file, test_file, out_path, label_name, emb_name, use_emb, args.emb_file,
                           is_emb_bin, freeze_emb, emb_dim, normalize_emb, is_binary_class, k_fold)
model_setup.config_train_val_test(dataset_name, model_type, use_uni_bi)
model_setup.build_vocs(dataset_name)
model_setup.build_iterators(batch_size, dataset_name)

# configure model
if model_type == "CNN":
	model_setup.config_CNN_model(model_type, num_filters, filter_sizes, dropout)
elif model_type == "LSTM":
	model_setup.config_RNN_model(model_type, hid_dim, is_bidirectional, n_layers, dropout)
elif model_type == "FastText":
	model_setup.config_FastText_model(model_type, dropout)
elif model_type == "SeqVecNet":
	model_setup.config_SeqVecNet_model(model_type, hid_dim, dropout)
elif model_type == "SeqVecCharNet":
	model_setup.config_SeqVecCharNet_model(model_type, hid_dim, dropout)

# configure loss + optimizer
# if (dataset_name == "DeepLoc" and label_name == "cellular_location") or dataset_name in ["TargetP", "NEW"]:
model_setup.calculate_label_weights("balanced_scikit")
# model_setup.calculate_label_weights("default")
model_setup.config_criterion_optimizer(learning_rate, weight_decay)

# --- Run model ---#
model_out_path = out_path
nn_model = NeurNet(model_setup, out_path, save_predictions)
nn_model.train_eval(num_epoches)
