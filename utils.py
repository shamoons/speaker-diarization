# utils.py
import torch
import argparse


def save_checkpoint(state, filepath):
    torch.save(state, filepath)


def load_checkpoint(filepath, model, optimizer=None):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint['epoch'], checkpoint['train_loss']


def get_arg_parser():
    parser = argparse.ArgumentParser(description='Transformer Autoencoder for SpeechCommands Dataset')
    parser.add_argument('--data_path', type=str, default='./data', help='Path to the dataset')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--lite', type=int, default=None, help='Use a lite version of the dataset')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--checkpoint_path', type=str, default='', help='Path to the checkpoint file')
    parser.add_argument('--use_cuda', action=argparse.BooleanOptionalAction, help='Use CUDA if available')
    parser.add_argument('--use_mps', action=argparse.BooleanOptionalAction, help='Use MPS if available')
    parser.add_argument('--dropout', type=float, default=0, help='Dropout probability')

    # Transformer configuration
    parser.add_argument('--d_model', type=int, default=40, help='Number of expected features in the input')
    parser.add_argument('--nhead', type=int, default=2, help='Number of heads in the multihead attention models')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of sub-encoder-layers in the transformer encoder')
    parser.add_argument('--dim_feedforward', type=int, default=1024, help='Dimension of the feedforward network model')

    # Data configuration
    parser.add_argument('--feature_type', type=str, default='melspectrogram',
                        choices=['melspectrogram', 'mfcc'], help='Feature type to use: "melspectrogram" or "mfcc"')

    return parser
