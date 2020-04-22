import torch.nn as nn
import torch.nn.functional as F

"""
### SeqVecNet ###
Network that uses SeqVec embeddings to infer
as in Figure 1 (right) of
Heinzinger, Michael, et al. "Modeling the Language of Life-Deep Learning Protein Sequences." bioRxiv (2019): 614313.
"""

class SeqVecCharNet(nn.Module):

	def __init__(self,vocab_size,embedding_dim,hid_dim,output_dim,dropout,pad_idx):
		super().__init__()

		self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
		self.fc1 = nn.Linear(embedding_dim, hid_dim)
		self.dropout = nn.Dropout(dropout)
		self.bn = nn.BatchNorm1d(hid_dim)
		self.fc2 = nn.Linear(hid_dim, output_dim)

	def forward(self,text):
		# each text instance has a whole protein sequence as word (one word sentence)
		# text = [seq feature len, batch size]
		embedded = self.embedding(text)

		# embedded = [sent len, batch_size, emb dim]
		
		squeezed = embedded.squeeze(0)
		
		# squeezed = [batch_size, emb dim]

		reduced = self.fc1(squeezed)

		# reduced = [batch_size, hid_dim]

		filtered = F.relu(self.dropout(reduced))

		# filtered = [batch_size, hid_dim]

		if filtered.shape[0] == 1:
			normalized = filtered
		else:
			normalized = self.bn(filtered)

		# normalized = [batch_size, hid_dim]
		out = self.fc2(normalized)
		return out

