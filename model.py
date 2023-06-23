# model.py
import torch
from torch import nn
import math


class SpeakerIdentificationModel(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, num_classes, input_size, dropout=0.1):
        super(SpeakerIdentificationModel, self).__init__()

        self.d_model = d_model

        # Input embedding
        self.input_embedding = nn.Linear(input_size, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        # Classifier
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, src):
        # Apply the input embedding
        src = self.input_embedding(src)  # (batch_size, sequence_length, d_model)

        src = src.transpose(0, 1)  # (sequence_length, batch_size, d_model)

        # Add positional encoding to the input
        src = self.pos_encoder(src)  # (sequence_length, batch_size, d_model)

        # Pass the input through the Transformer encoder
        output = self.transformer_encoder(src)  # (sequence_length, batch_size, d_model)

        # Average pooling over the sequence dimension
        output = output.mean(dim=0)  # (batch_size, d_model)

        # Pass the output of the transformer through the classifier
        output = self.classifier(output)  # (batch_size, num_classes)

        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: tensor of shape (sequence_length, batch_size, d_model)

        x = x + self.pe[:x.size(0), :]  # (sequence_length, batch_size, d_model)

        return self.dropout(x)
