import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
	"""
	Class to implement CNN2d for text
	Credits: https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/4%20-%20Convolutional%20Sentiment%20Analysis.ipynb
	"""
	def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim,
	             dropout, pad_idx):
		super().__init__()
		self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

		self.convs = nn.ModuleList([
			nn.Conv2d(in_channels=1,
			          out_channels=n_filters,
			          kernel_size=(fs, embedding_dim))
			for fs in filter_sizes
		])

		# self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
		#mutli-layers
		self.fc = nn.Linear(len(filter_sizes) * n_filters, 128)
		self.fc2 = nn.Linear(128, 128)
		self.fc3 = nn.Linear(128, output_dim)
		self.dropout = nn.Dropout(dropout)
		self.bn = nn.BatchNorm1d(n_filters * len(filter_sizes))


	def forward(self, text):
		# text = [sent len, batch size]

		text = text.permute(1, 0)

		# text = [batch size, sent len]

		embedded = self.embedding(text)

		# embedded = [batch size, sent len, emb dim]

		embedded = embedded.unsqueeze(1)

		# embedded = [batch size, 1, sent len, emb dim]

		conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]

		# conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]

		pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

		# pooled_n = [batch size, n_filters]

		cat = self.dropout(torch.cat(pooled, dim=1))
		# cat = torch.cat(pooled, dim=1)
		# cat = [batch size, n_filters * len(filter_sizes)]

		normalized = self.bn(cat)

		# normalized = [batch size, n_filters * len(filter_sizes)]
		# one layer
		# return self.fc(normalized)

		#multi-layers
		# fc_out = F.relu(self.dropout(self.fc(normalized)))
		fc_out = F.relu(self.fc(normalized))
		# return self.fc2(fc_out)
		fc2_out = F.relu(self.fc2(fc_out))
		return self.fc3(fc2_out)