import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN1d(nn.Module):
	"""
	Credits: https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/4%20-%20Convolutional%20Sentiment%20Analysis.ipynb
	"""

	def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout, pad_idx):
		print("CNN1d")
		super().__init__()

		self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

		self.convs = nn.ModuleList([
			nn.Conv1d(in_channels=embedding_dim,
			          out_channels=n_filters,
			          kernel_size=fs)
			for fs in filter_sizes
		])

		self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
		self.bn = nn.BatchNorm1d(output_dim)
		self.dropout = nn.Dropout(dropout)

	def forward(self, text):
		# text = [sent len, batch size]

		text = text.permute(1, 0)

		# text = [batch size, sent len]

		embedded = self.embedding(text)

		# embedded = [batch size, sent len, emb dim]

		embedded = embedded.permute(0, 2, 1)

		# embedded = [batch size, emb dim, sent len]

		conved = [F.relu(conv(embedded)) for conv in self.convs]

		# conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]

		pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

		# pooled_n = [batch size, n_filters]

		cat = self.dropout(torch.cat(pooled, dim=1))

		# cat = [batch size, n_filters * len(filter_sizes)]
		if cat.shape[0] == 1:
			return self.fc(cat)
		else:
			return self.bn(self.fc(cat))
