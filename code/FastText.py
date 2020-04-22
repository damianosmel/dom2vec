import torch.nn as nn
import torch.nn.functional as F


class FastText(nn.Module):
	def __init__(self, vocab_size, embedding_dim, output_dim, dropout, pad_idx):
		super().__init__()

		self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

		self.bn = nn.BatchNorm1d(output_dim)
		self.dropout = nn.Dropout(dropout)
		self.fc = nn.Linear(embedding_dim, output_dim)

	def forward(self, text):
		# text = [sent len, batch size]

		embedded = self.embedding(text)

		# embedded = [sent len, batch size, emb dim]

		embedded = embedded.permute(1, 0, 2)

		# embedded = [batch size, sent len, emb dim]

		pooled = F.avg_pool2d(embedded, (embedded.shape[1], 1)).squeeze(1)

		# pooled = [batch size, embedding_dim]

		if pooled.shape[0] == 1:
			batch_norm = self.dropout(self.fc(pooled))
		else:
			batch_norm = self.bn(self.dropout(self.fc(pooled)))

		# batch_norm = [batch_size, embedding_dim]
		return batch_norm
