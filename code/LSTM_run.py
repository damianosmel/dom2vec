import torch
import torch.nn as nn
import torch.optim as optim
import torchtext.vocab as vocab
from gensim.models import KeyedVectors
from torchtext import data
import random
from RNN import RNN
from utils import sec2hour_min_sec
from timeit import default_timer as timer
import os
from ntpath import basename


###                                ###
### Deprecated Class               ###
### please see neurNet_run instead ###
###                                ###

###
# Running one hot -> embedding -> RNN -> fully connected layer -> prediction
# Credits: http://anie.me/On-Torchtext/
# https://github.com/bentrevett/pytorch-sentiment-analysis
###

def convert_bin_emb_txt(out_path, emb_file):
	txt_name = basename(emb_file).split(".")[0] + ".txt"
	emb_txt_file = os.path.join(out_path, txt_name)
	emb_model = KeyedVectors.load_word2vec_format(emb_file, binary=True)
	emb_model.save_word2vec_format(emb_txt_file, binary=False)
	return emb_txt_file


def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)


def binary_accuracy(preds, y):
	"""
	Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
	"""

	# round predictions to the closest integer
	rounded_preds = torch.round(torch.sigmoid(preds))
	correct = (rounded_preds == y).float()  # convert into float for division
	acc = correct.sum() / len(correct)
	return acc


def categorical_accuracy(preds, y):
	"""
	Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
	"""
	max_preds = preds.argmax(dim=1, keepdim=True)  # get the index of the max probability
	correct = max_preds.squeeze(1).eq(y)
	return correct.sum() / torch.FloatTensor([y.shape[0]])


def train(model, iterator, optimizer, criterion):
	epoch_loss = 0
	epoch_acc = 0

	model.train()

	for batch in iterator:
		optimizer.zero_grad()
		text, text_lengths = batch.domains  # add text length
		# predictions = model(batch.domains).squeeze(1)#binary
		predictions = model(text, text_lengths)
		loss = criterion(predictions, batch.label)
		# acc = binary_accuracy(predictions, batch.label)#binary
		acc = categorical_accuracy(predictions, batch.label)
		loss.backward()
		optimizer.step()

		epoch_loss += loss.item()
		epoch_acc += acc.item()

	return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
	epoch_loss = 0
	epoch_acc = 0

	model.eval()

	with torch.no_grad():
		for batch in iterator:
			# predictions = model(batch.domains).squeeze(1)#binary
			text, text_lengths = batch.domains  # add text length
			predictions = model(text, text_lengths)
			loss = criterion(predictions, batch.label)
			# acc = binary_accuracy(predictions, batch.label)#binary
			acc = categorical_accuracy(predictions, batch.label)
			epoch_loss += loss.item()
			epoch_acc += acc.item()

	return epoch_loss / len(iterator), epoch_acc / len(iterator)


SEED = 1234
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

data_path = "/home/damian/Documents/L3S/projects/datasets/deeploc"
data_train_file = "deeploc_dataset_train.csv"
data_test_file = "deeploc_dataset_test.csv"
out_path = "/home/damian/Documents/L3S/projects/datasets/deeploc"  # OUT_PATH should be different for each run in server
label_name = "memb_soluble"  # "cell_location"#

use_emb = True
# emb_file = "/home/damian/Documents/L3S/projects/overlap/dom2vec_sg_w2_d50_e15/dom2vec_w2_cbow0_hierSoft0_dim50_e10.bin"
emb_file = "/home/damian/Documents/L3S/projects/datasets/deeploc/dom2vec_w2_cbow0_hierSoft0_dim50_e10.txt"
is_emb_bin = False
if is_emb_bin:
	emb_file = convert_bin_emb_txt(out_path, emb_file)

###
# hyperparameters
###
n_layers = 1
"""
	DeepLoc:
	train_test cellular_location membrane_soluble interpro_domains
	##
	# csv: [ 'train_test'   'cellular_location' 'membrane_soluble' 'interpro_domains']
	# use: [  train_test    cel_loc (label)      mem_sol (label)             domains]
	##
"""
###
# Labels
###
if label_name == "cell_location":
	TRAIN_TEST = data.Field(dtype=torch.float)
	# LABEL = data.LabelField(dtype=torch.float) #binary
	LABEL = data.LabelField()  # multi-class
	MEMBR_SOL = data.Field(dtype=torch.float)
	INTERPRO_DOMAINS = data.Field(sequential=True, include_lengths=True)  # split on spaces
	fields = [('train_test', TRAIN_TEST), ('label', LABEL), ('mem_sol', MEMBR_SOL), ('domains', INTERPRO_DOMAINS)]
else:
	TRAIN_TEST = data.Field(dtype=torch.float)
	CELLULAR_LOC = data.Field(dtype=torch.float)
	# LABEL = data.LabelField(dtype=torch.float)#binary
	LABEL = data.LabelField()  # multi-class
	INTERPRO_DOMAINS = data.Field(sequential=True, include_lengths=True)  # split on spaces
	fields = [('train_test', TRAIN_TEST), ('cel_loc', CELLULAR_LOC), ('label', LABEL), ('domains', INTERPRO_DOMAINS)]

train_data, test_data = data.TabularDataset.splits(
	path=data_path,
	train=data_train_file,
	test=data_test_file,
	format='csv',
	fields=fields,
	skip_header=True
)
###
# Get validation set
###
train_data, valid_data = train_data.split(random_state=random.seed(SEED))

print("=== Stats of examples ===")
print('Number of training examples: {}'.format(len(train_data)))
print('Number of validation examples: {}'.format(len(valid_data)))
print('Number of testing examples: {}'.format(len(test_data)))
# print("Values of the first example: {}".format(vars(train_data.examples[0])))

MAX_VOCAB_SIZE = 35_000
if use_emb:
	print("Loading embeddings from: {}".format(emb_file))
	custom_embeddings = vocab.Vectors(name=emb_file,
	                                  cache=os.path.join(out_path, "custom_embeddings"),
	                                  unk_init=torch.Tensor.normal_)

	INTERPRO_DOMAINS.build_vocab(train_data,
	                             max_size=MAX_VOCAB_SIZE,
	                             vectors=custom_embeddings,
	                             unk_init=torch.Tensor.normal_)
else:
	INTERPRO_DOMAINS.build_vocab(train_data, max_size=MAX_VOCAB_SIZE)

if label_name == "cell_location":
	MEMBR_SOL.build_vocab(train_data)
else:
	CELLULAR_LOC.build_vocab(train_data)
LABEL.build_vocab(train_data)
TRAIN_TEST.build_vocab(train_data)
print("=== Stats for variables ===")
print("Number of unique domains: {}".format(len(INTERPRO_DOMAINS.vocab)))
print("Number of labels: {}".format(len(LABEL.vocab)))
# print("Labels index: {}".format(LABEL.vocab.stoi))

###
# Iterators
###
BATCH_SIZE = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
	(train_data, valid_data, test_data),
	sort_key=lambda x: x.domains,
	batch_size=BATCH_SIZE,
	device=device)

###
# Model
###
INPUT_DIM = len(INTERPRO_DOMAINS.vocab)
EMBEDDING_DIM = 50
HIDDEN_DIM = 128
OUTPUT_DIM = len(LABEL.vocab)
N_LAYERS = n_layers
BIDIRECTIONAL = True
if N_LAYERS == 1:
	DROPOUT = 0.0
else:
	DROPOUT = 0.5
PAD_IDX = INTERPRO_DOMAINS.vocab.stoi[INTERPRO_DOMAINS.pad_token]

model = RNN(INPUT_DIM,
            EMBEDDING_DIM,
            HIDDEN_DIM,
            OUTPUT_DIM,
            N_LAYERS,
            BIDIRECTIONAL,
            DROPOUT,
            PAD_IDX)
print("The model has {} trainable parameters.".format(count_parameters(model)))
embeddings = INTERPRO_DOMAINS.vocab.vectors
print("Embeddings shape: {}".format(embeddings.shape))
print("Replace embeddings for known domains.")
model.embedding.weight.data.copy_(embeddings)
print("Set up embeddings of <unk> and <pad> to 0.")
UNK_IDX = INTERPRO_DOMAINS.vocab.stoi[INTERPRO_DOMAINS.unk_token]
PAD_IDX = INTERPRO_DOMAINS.vocab.stoi[INTERPRO_DOMAINS.pad_token]
model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
# print("Initial embeddings:\n {}".format(model.embedding.weight.data))
###
# Optimizer
###
# optimizer = optim.SGD(model.parameters(), lr=1e-3)
# criterion = nn.BCEWithLogitsLoss()#binary
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()
model = model.to(device)
criterion = criterion.to(device)

###
# Running model
###
N_EPOCHS = 250

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

	time_start = timer()
	train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
	valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
	time_end = timer()

	if valid_loss < best_valid_loss:
		best_valid_loss = valid_loss
		torch.save(model.state_dict(), os.path.join(out_path, "tut1-model.pt"))

	print("Epoch: {} | Epoch Time: {}".format(epoch + 1, sec2hour_min_sec(time_end - time_start)))
	print("\tTrain Loss: {:.3f} | Train Acc: {:.2f}".format(train_loss, train_acc * 100))
	print("\t Val. Loss: {:.3f} |  Val. Acc: {:.2f}".format(valid_loss, valid_acc * 100))

###
# Test accuracy
###
model.load_state_dict(torch.load(os.path.join(out_path, "tut1-model.pt")))
test_loss, test_acc = evaluate(model, test_iterator, criterion)
print("Test Loss: {:.3f} | Test Acc. Acc: {:.2f}".format(test_loss, test_acc * 100))
# print(f'Test Loss: {:.3f} | Test Acc: {:.2f}'.format(test_loss,test_acc*100))
