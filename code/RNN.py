import torch
import torch.nn as nn


class RNN(nn.Module):
	"""
	Class to impelement RNN
	Credits: https://github.com/bentrevett/pytorch-sentiment-analysis
	"""

	def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,
	             bidirectional, dropout, pad_idx):
		super().__init__()
		self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

		self.rnn = nn.LSTM(embedding_dim,
		                   hidden_dim,
		                   num_layers=n_layers,
		                   bidirectional=bidirectional,
		                   dropout=dropout)
		if bidirectional:
			self.fc = nn.Linear(hidden_dim * 2, output_dim)
		else:
			self.fc = nn.Linear(hidden_dim, output_dim)
		self.is_bidirectional = bidirectional

		self.bn = nn.BatchNorm1d(output_dim)
		self.dropout = nn.Dropout(dropout)

	def forward(self, text, text_lengths):
		# text = [sent len, batch size]

		embedded = self.dropout(self.embedding(text))
		# embedded = [sent len, batch size, emb dim]

		# pack sequence
		packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, enforce_sorted=False)

		packed_output, (hidden, cell) = self.rnn(packed_embedded)

		# unpack sequence
		output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)

		# output = [sent len, batch size, hid dim * num directions]
		# output over padding tokens are zero tensors

		# hidden = [num layers * num directions, batch size, hid dim]
		# cell = [num layers * num directions, batch size, hid dim]

		# concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
		# and apply dropout
		if self.is_bidirectional:
			hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
		else:
			hidden = self.dropout(hidden[0, :, :])
		# hidden = [batch size, hid dim * num directions]

		if hidden.shape[0] == 1:
			return self.fc(hidden)
		else:
			return self.bn(self.fc(hidden))
