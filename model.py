# model.py
import torch
from torch import nn

from positional_encoding import PositionalEncoding


class SpeakerIdentificationModel(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, num_classes, n_time_steps, dropout=0.1):
        super(SpeakerIdentificationModel, self).__init__()

        self.d_model = d_model
        self.n_time_steps = n_time_steps

        # Input embedding: Convert the input of shape (batch_size * new_sequence_length, n_time_steps * d_model)
        # to (batch_size * new_sequence_length, d_model)
        self.input_embedding = nn.Linear(n_time_steps * d_model, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Classifier: Convert the output of shape (batch_size, d_model) to (batch_size, num_classes)
        self.classifier = nn.Linear(d_model, num_classes)

        # Softmax for the final output
        self.softmax = nn.Softmax(dim=1)

    def forward(self, src):
        # src is a tensor of shape (batch_size, sequence_length, n_mels)
        batch_size, sequence_length, n_mels = src.shape  # get the original shape

        new_sequence_length = sequence_length // self.n_time_steps
        src = src.unfold(1, self.n_time_steps, self.n_time_steps)  # unfold along the time axis

        # Reshape src to 2D for the linear layer (flatten the last two dimensions)
        # Now, src is of shape (batch_size * new_sequence_length, n_time_steps * d_model)
        src = src.reshape(batch_size * new_sequence_length, -1)

        # Apply the input embedding
        # Now, src is of shape (batch_size * new_sequence_length, d_model)
        src = self.input_embedding(src)

        # Reshape back to 3D format
        # Now, src is of shape (batch_size, new_sequence_length, d_model)
        src = src.view(batch_size, new_sequence_length, self.d_model)

        src = src.transpose(0, 1)  # (new_sequence_length, batch_size, d_model)

        # Add positional encoding to the input
        # Now, src is of shape (new_sequence_length, batch_size, d_model)
        src = self.pos_encoder(src)

        # Pass the input through the Transformer encoder
        # Now, output is of shape (new_sequence_length, batch_size, d_model)
        output = self.transformer_encoder(src)

        # Apply dropout
        output = self.dropout(output)

        # Average pooling over the sequence dimension
        # Now, output is of shape (batch_size, d_model)
        output = output.mean(dim=0)

        # Pass the output of the transformer through the classifier
        # Now, output is of shape (batch_size, num_classes)
        output = self.classifier(output)

        # Apply softmax to the output
        # Now, output is of shape (batch_size, num_classes)
        output = self.softmax(output)

        return output
