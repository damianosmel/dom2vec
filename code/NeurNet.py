import torch

from utils import sec2hour_min_sec
from timeit import default_timer as timer
import os
from time import localtime, strftime
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
import sklearn
import numpy as np

class NeurNet:
	"""
	Class to create the neural network
	Credits: https://github.com/bentrevett/pytorch-sentiment-analysis
	"""

	def __init__(self, configured_model, out_path, save_predictions):
		self.configured_model = configured_model
		self.models = configured_model.models
		self.model_type = configured_model.model_type
		self.k_fold = self.configured_model.k_fold
		self.batch_size = configured_model.batch_size
		self.learning_rate = configured_model.learning_rate
		if self.k_fold == -1:  # train,test
			self.train_iterator = configured_model.train_iterator
			self.valid_iterator = configured_model.valid_iterator
		elif self.k_fold == 0:  # train,eval
			self.train_iterator = configured_model.train_iterator
			self.valid_iterator = configured_model.valid_iterator
		else:  # k-fold
			self.fold_iterators = configured_model.fold_iterators
		self.test_iterator = configured_model.test_iterator
		self.emb_name = configured_model.emb_name
		self.optimizers = configured_model.optimizers
		self.criteria = configured_model.criteria
		self.is_binary_class = configured_model.is_binary_class

		self.out_path = out_path
		self.save_predictions = save_predictions

	def binary_accuracy(self, preds, y):
		"""
		Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
		"""
		# round predictions to the closest integer
		rounded_preds = torch.round(torch.sigmoid(preds))
		correct = (rounded_preds == y).float()  # convert into float for division
		acc = correct.sum() / len(correct)
		return acc

	def precision(self, preds, y):
		if self.is_binary_class:
			# round predictions to the closest integer
			final_preds = torch.round(torch.sigmoid(preds))
		else:
			final_preds = preds.argmax(dim=1, keepdim=True)  # get the index of the max probability
			final_preds = final_preds.squeeze(1)
		return precision_score(y.detach().cpu().numpy(), final_preds.detach().cpu().numpy(), average='macro')

	def recall(self, preds, y):
		if self.is_binary_class:
			final_preds = torch.round(torch.sigmoid(preds))
		else:
			final_preds = preds.argmax(dim=1, keepdim=True)
			final_preds = final_preds.squeeze(1)
		return recall_score(y.detach().cpu().numpy(), final_preds.detach().cpu().numpy(), average='macro')

	def auroc(self, preds, y):
		# print("sklean ver: {}".format(sklearn.__version__))
		if self.is_binary_class:
			final_preds = torch.sigmoid(preds)
			# print("y={}".format(y))
			# print("final_preds={}".format(final_preds))
			if torch.sum(y) == self.batch_size:
				print("All instances are positive so AuROC is 0")
				return 0.0
			elif torch.sum(y) == 0:
				print("All instances are negative so AUROC is 0")
				return 0.0
			else:
				return roc_auc_score(y.detach().cpu().numpy(), final_preds.detach().cpu().numpy())
		else:
			final_preds = torch.softmax(preds, dim=1).squeeze(1)
			num_classes = final_preds.shape[1]
			label_indices = np.arange(num_classes)
			return roc_auc_score(y.detach().cpu().numpy(), final_preds.detach().cpu().numpy(), average='macro', multi_class='ovo',labels=label_indices)


	def f1_prec_rec(self, recall, precision):
		if precision == recall == 0:
			return 0.0
		else:
			return 2.0 * (precision * recall) / (precision + recall)

	def balanced_bin_accuracy(self, preds, y):
		rounded_preds = torch.round(torch.sigmoid(preds))
		return balanced_accuracy_score(y.detach().cpu().numpy(), rounded_preds.detach().cpu().numpy())

	def balanced_accuracy(self, preds, y):
		"""
		Returns the balanced accuracy as explained at:
		https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html#sklearn.metrics.balanced_accuracy_score
		:param preds: predictions
		:param y: true labels
		:return: balanced_accuracy = 1/sum(w_i) * sum(I(pred_i=y_i)*w_i)
		"""
		max_preds = preds.argmax(dim=1, keepdim=True).squeeze(1)  # get the index of the max probability
		return balanced_accuracy_score(y.detach().cpu().numpy(), max_preds.detach().cpu().numpy())

	def categorical_accuracy(self, preds, y):
		"""
		Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
		"""
		max_preds = preds.argmax(dim=1, keepdim=True)  # get the index of the max probability

		correct = max_preds.squeeze(1).eq(y)
		return correct.sum() / torch.FloatTensor([y.shape[0]])

	def model_train(self, batches, model_idx):
		epoch_loss = 0
		epoch_acc = 0

		self.models[model_idx].train()
		for batch in batches:
			self.optimizers[model_idx].zero_grad()
			# add text length
			if self.emb_name == "dom2vec":
				text, text_lengths = batch.domains
			else:  # seqvec, protvec
				text, text_lengths = batch.seq
			if self.is_binary_class:  # binary
				if self.model_type == "LSTM":
					predictions = self.models[model_idx](text, text_lengths).squeeze(1)
				elif self.model_type == "CNN" or self.model_type == "FastText" or self.model_type == "SeqVecNet" or self.model_type == "SeqVecCharNet":
					predictions = self.models[model_idx](text).squeeze(1)
			else:  # multi-class
				if self.model_type == "LSTM":
					predictions = self.models[model_idx](text, text_lengths)
				elif self.model_type == "CNN" or self.model_type == "FastText" or self.model_type == "SeqVecNet" or self.model_type == "SeqVecCharNet":
					predictions = self.models[model_idx](text)

			loss = self.criteria[model_idx](predictions, batch.label)
			if self.is_binary_class:
				acc = self.binary_accuracy(predictions, batch.label)  # binary
			else:
				acc = self.categorical_accuracy(predictions, batch.label)
			loss.backward()
			self.optimizers[model_idx].step()

			epoch_loss += loss.item()
			epoch_acc += acc.item()

		return epoch_loss / len(batches), epoch_acc / len(batches)

	def get_predictions(self, batches, model_idx):
		#credits: https://stackoverflow.com/questions/48264368/save-predictions-from-pytorch-model
		y_true, y_predicted = [], []
		self.models[model_idx].eval()

		with torch.no_grad():
			for batch in batches:
				# add text length
				if self.emb_name == "dom2vec":
					text, text_lengths = batch.domains
				else: #seqvec, protvec
					text, text_lengths = batch.seq
				if self.is_binary_class: #binary
					if self.model_type == "LSTM":
						predictions = self.models[model_idx](text, text_lengths).squeeze(1)
					elif self.model_type == "CNN" or self.model_type == "FastText" or self.model_type == "SeqVecNet" or self.model_type == "SeqVecCharNet":
						predictions = self.models[model_idx](text).squeeze(1)
					y_predicted = y_predicted + torch.round(torch.sigmoid(predictions)).data.cpu().tolist()
				else: #multi-class
					if self.model_type == "LSTM":
						predictions = self.models[model_idx](text,text_lengths)
					elif self.model_type == "CNN" or self.model_type == "FastText" or self.model_type == "SeqVecNet" or self.model_type == "SeqVecCharNet":
						predictions = self.models[model_idx](text)
					y_predicted = y_predicted + predictions.argmax(dim=1, keepdim=True).squeeze(1).data.cpu().tolist()
				y_true = y_true + batch.label.data.cpu().tolist()
		assert len(y_predicted) == len(y_true), "AssertionError: the predicted and true labels are not of equal size."
		return y_true, y_predicted

	def write_predictions(self, y_true, y_predicted):
		out_name = "best_test_predictions.csv"
		with open(os.path.join(self.out_path, out_name), 'w') as file_out:
			file_out.write("y_true,y_predicted\n")
			for i in range(len(y_true)):
				file_out.write(str(y_true[i]) + "," + str(y_predicted[i]) + "\n")
		print("Test set predictions of best model are saved at {}".format(os.path.join(self.out_path,out_name)))

	def model_evaluate(self, batches, model_idx):
		epoch_loss = 0
		epoch_acc = 0
		epoch_auroc = 0

		self.models[model_idx].eval()
		with torch.no_grad():
			for batch in batches:
				# add text length
				if self.emb_name == "dom2vec":
					text, text_lengths = batch.domains
				else:  # seqvec, protvec
					text, text_lengths = batch.seq
				if self.is_binary_class:  # binary
					if self.model_type == "LSTM":
						predictions = self.models[model_idx](text, text_lengths).squeeze(1)
					elif self.model_type == "CNN" or self.model_type == "FastText" or self.model_type == "SeqVecNet" or self.model_type == "SeqVecCharNet":
						predictions = self.models[model_idx](text).squeeze(1)
				else:  # multi-class
					if self.model_type == "LSTM":
						predictions = self.models[model_idx](text, text_lengths)
					elif self.model_type == "CNN" or self.model_type == "FastText" or self.model_type == "SeqVecNet" or self.model_type == "SeqVecCharNet":
						predictions = self.models[model_idx](text)
				loss = self.criteria[model_idx](predictions, batch.label)
				if self.is_binary_class:
					acc = self.binary_accuracy(predictions, batch.label)  # binary
				else:
					acc = self.categorical_accuracy(predictions, batch.label)
				auroc = self.auroc(predictions, batch.label)
				epoch_loss += loss.item()
				epoch_acc += acc.item()
				epoch_auroc += auroc #auroc.item()
		return epoch_loss/len(batches), epoch_acc/len(batches), epoch_auroc/len(batches)

	def train_eval(self, num_epoches):
		if self.k_fold == -1:  # train, test
			self.train_eval_test(num_epoches)
		elif self.k_fold == 0:  # train, val
			self.train_eval_val(num_epoches)
		else:  # train, val k-fold
			self.train_eval_kfold(num_epoches)

	def train_eval_test(self, num_epoches):
		###
		# Running model for train on train set
		# when evaluation on validation is better than before
		# evaluate on test set
		###
		N_EPOCHES = num_epoches
		test_period = 50
		best_train_loss, best_val_loss, best_test_loss = float('inf'), float('inf'), float('inf')
		best_train_acc, best_val_acc, best_test_acc = -float('inf'), -float('inf'), -float('inf')
		best_test_auroc = -float('inf')
		y_true, y_predicted = [], []
		model_idx = 0
		total_time = 0
		for epoch in range(N_EPOCHES):
			time_start = timer()
			train_loss, train_acc = self.model_train(self.train_iterator, model_idx)
			time_end = timer()
			total_time = total_time + (time_end - time_start)
			val_loss, val_acc, val_auroc = self.model_evaluate(self.valid_iterator, model_idx)
			if (epoch + 1) % test_period == 0:
				print("Epoch: {}".format(epoch + 1))
				print("\tTrain Loss: {:.3f} | Train Acc: {:.4f}".format(train_loss, train_acc * 100))
				print("\tVal Loss: {:.3f} | Val Acc: {:.4f}".format(val_loss, val_acc * 100))
			if val_loss < best_val_loss:
				best_val_loss = val_loss
				test_loss, test_acc, test_auroc = self.model_evaluate(self.test_iterator, model_idx)
				print("\tTest Loss: {:.3f} | Test Acc: {:.4f} | Test AuROC: {:.4f}".format(test_loss, test_acc * 100,test_auroc))
				if train_acc > best_train_acc:
					best_train_acc = train_acc
				if val_acc > best_val_acc:
					best_val_acc = val_acc
				if test_acc > best_test_acc:
					best_test_acc = test_acc
				if test_auroc > best_test_auroc:
					best_test_auroc = test_auroc
					if self.save_predictions:
						y_true, y_predicted = self.get_predictions(self.test_iterator, model_idx)

		print("\n--- Run Best Acc ---")
		print("Train best acc: {:.4f}".format(best_train_acc * 100))
		print("Val best acc: {:.4f}".format(best_val_acc * 100))
		print("Test best acc: {:.4f} | best AuROC: {:.4f}".format(best_test_acc*100, best_test_auroc))
		print("Total train time: {}".format(sec2hour_min_sec(total_time)))
		print("Average train time: {}".format(sec2hour_min_sec(total_time/N_EPOCHES)))
		if self.save_predictions:
			self.write_predictions(y_true,y_predicted)

	def train_eval_val(self, num_epoches):
		###
		# Running model for train and test splits
		###
		N_EPOCHS = num_epoches
		ep_last_test_check = 20
		model_idx = 0
		best_valid_acc = -float('inf')
		train_losses = []
		valid_losses = []
		test_losses = []
		for epoch in range(N_EPOCHS):

			time_start = timer()
			train_loss, train_acc = self.model_train(self.train_iterator, model_idx)
			valid_loss, valid_acc, valid_auroc = self.model_evaluate(self.valid_iterator, model_idx)
			test_loss, test_acc, test_auroc = self.model_evaluate(self.test_iterator, model_idx)
			time_end = timer()
			if (epoch + 1) % ep_last_test_check == 0:
				print("Epoch: {}".format(epoch + 1))
				print("\tTrain Loss: {:.3f} | Train Acc: {:.4f}".format(train_loss, train_acc * 100))
				print("\tVal. Loss: {:.3f} | Val. Acc: {:.4f} | Val. AuROC: {:.4f}".format(valid_loss, valid_acc * 100, valid_auroc))
				print("\tTest Loss: {:.3f} | Test Acc: {:.4f} | Val. AuROC: {:.4f}".format(test_loss, test_acc * 100, test_auroc))

			train_losses.append(train_loss)
			valid_losses.append(valid_loss)
			test_losses.append(test_loss)
		print("Ploting train/validation loss")
		self.plot_losses(train_losses, valid_losses, test_losses)

	def plot_losses(self, train_losses, valid_losses, test_losses):
		fig = plt.figure()
		plt.plot(train_losses, label='Training loss')
		plt.plot(valid_losses, label='Validation loss')
		plt.plot(test_losses, label='Test loss')
		plt.title("Train/Val/Test Loss")
		plt.legend(frameon=False)
		t = localtime()
		timestamp = strftime('%b-%d-%Y_%H%M', t)
		plot_name = self.model_type + '_b' + str(self.batch_size) + "_lr" + str(
			self.learning_rate) + "_" + timestamp + ".png"
		fig.savefig(os.path.join(self.out_path, plot_name), bbox_inches='tight', dpi=600)

	def train_eval_kfold(self, num_epoches):
		###
		# Running model for k-fold
		###
		N_EPOCHS = num_epoches
		folds_train_acc = []
		folds_val_acc = []

		folds_val_auroc = []
		num_epoches_print = 50

		for i in range(0, self.k_fold):
			best_val_loss = float('inf')
			best_val_acc = 0.0
			best_train_acc = 0.0
			best_val_auroc = 0.0
			print("\n--- Fold {} ---".format(i))
			for epoch in range(N_EPOCHS):
				time_start = timer()
				train_loss, train_acc = self.model_train(self.fold_iterators[i][0], i)
				val_loss, val_acc, val_auroc = self.model_evaluate(self.fold_iterators[i][1],i)
				time_end = timer()
				if (epoch + 1) % num_epoches_print == 0:
					print("Epoch: {} | Epoch Time: {}".format(epoch + 1, sec2hour_min_sec(time_end - time_start)))
					print("\tTrain Loss: {:.3f} | Train Acc: {:.4f}".format(train_loss, train_acc * 100))
					print("\t Val. Loss: {:.3f} |  Val. Acc: {:.4f}".format(val_loss, val_acc * 100))
					print("\t Val. AuROC: {:.4f}".format(val_auroc))

				if val_loss < best_val_loss:
					best_val_loss = val_loss
					if val_auroc > best_val_auroc:
						best_val_auroc = val_auroc
					if val_acc > best_val_acc:
						best_val_acc = val_acc
					if train_acc > best_train_acc:
						best_train_acc = train_acc

			folds_train_acc.append(best_train_acc)
			folds_val_acc.append(best_val_acc)
			folds_val_auroc.append(best_val_auroc)

		print("\n--- Average over folds ---")
		avg_val_auroc = sum(folds_val_auroc) / self.k_fold
		print("Average AuROC over all folds: {:.4f}".format(avg_val_auroc))
