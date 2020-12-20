import torch
import torch.nn as nn
import torch.nn.functional as F


class AdvertisementGenerator(nn.Module):
    def __init__(
        self, 
        num_embeddings, 
        num_categories, 
        dim_embedding,
        dim_hidden, 
        num_layers=2, 
        dropout=0.1
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, dim_embedding)
        self.category_embedding = nn.Embedding(num_categories, dim_embedding)

        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.LSTM(dim_embedding, dim_hidden, num_layers, dropout=(dropout if num_layers > 1 else 0), batch_first=True)
        self.fc = nn.Linear(2*dim_hidden, num_embeddings)
        self.fc_memory = nn.Linear(dim_embedding, dim_hidden)

    def forward(self, txts, categories, keyphrases, keyphrases_mask=None, keyphrases_padding_mask=None):
        # Encode keyphrases
        memory = self.encode_keyphrase(keyphrases, keyphrases_padding_mask)

        embedded = self.dropout(self.embedding(txts))
        embedded[:, 0, :] = self.dropout(self.category_embedding(categories)) # Use category label as begginin of sequence
        output, state = self.decoder(embedded)

        # Attend keyphrases
        context, _ = self.attention(output, memory, memory, keyphrases_mask)

        output = self.fc(self.dropout(torch.cat([output, context], dim=-1)))
        return output, state

    def encode_keyphrase(self, keyphrases, keyphrases_padding_mask=None):
        embedded = self.dropout(self.embedding(keyphrases))
        if keyphrases_padding_mask is not None:
            embedded[keyphrases_padding_mask] = 0.
            ntokens = torch.sum(~keyphrases_padding_mask, dim=2, keepdim=True)
            ntokens[ntokens == 0] = 1 # To avoid zero division
            output = torch.sum(embedded, dim=2) / ntokens.type_as(embedded)
        else:
            output = torch.mean(embedded, dim=2)

        return self.fc_memory(output)

    def attention(self, query, key, value, mask=None):
        weights = torch.matmul(query, key.transpose(-2, -1))
        if mask is not None:
            weights = weights.masked_fill(mask.unsqueeze(1), -1e9)

        weights = F.softmax(weights, dim=-1)
        context = torch.matmul(weights, value)
        return context, weights
    
    def step(self, last_predictions, state, timestep):
        last_predictions = last_predictions.unsqueeze(1)
        categories = state['categories'].unsqueeze(1)
        memory = state['memory'] 
        keyphrases_mask = state['keyphrases_mask']
        hidden = state['hidden'].permute(1, 0, 2).contiguous() if 'hidden' in state else None
        cell = state['cell'].permute(1, 0, 2).contiguous() if 'cell' in state else None

        if timestep == 0:
            embedded = self.dropout(self.category_embedding(categories))
        else:
            embedded = self.dropout(self.embedding(last_predictions))
        output, (hidden, cell) = self.decoder(embedded, (hidden, cell) if hidden is not None else None)

        # Attend keyphrases
        context, _ = self.attention(output, memory, memory, keyphrases_mask)

        output = self.fc(self.dropout(torch.cat([output, context], dim=-1)))
        output = output.squeeze(1)
        log_probabilities = F.log_softmax(output, dim=-1)

        hidden = hidden.permute(1, 0, 2)
        cell = cell.permute(1, 0, 2)
        return log_probabilities, {'hidden': hidden, 'cell': cell,  'memory': memory, 'categories': categories, 'keyphrases_mask': keyphrases_mask}

