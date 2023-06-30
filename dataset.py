# dataset.py
import os
import pickle
import glob

import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
import torchaudio

# File path to save/load the LabelEncoder
encoder_filepath = "data/encoder.pkl"
audio_path = "data/audio"

encoder = LabelEncoder()

# Check if the encoder file exists
if os.path.isfile(encoder_filepath):
    # Load the encoder from file
    with open(encoder_filepath, "rb") as f:
        encoder = pickle.load(f)
else:
    # If encoder file does not exist, prepare it from the dataset.
    speaker_ids = [os.path.basename(f.rstrip(os.sep)) for f in glob.glob(f"{audio_path}/*/*/")]
    encoder.fit(speaker_ids)
    # Save the fitted encoder for future use
    with open(encoder_filepath, "wb") as f:
        pickle.dump(encoder, f)

# Print the number of unique speakers
print("Number of unique speakers:", len(encoder.classes_))


class AudioDataset(Dataset):
    def __init__(self, audio_dir, lite=None):
        self.audio_dir = audio_dir
        self.audio_files = []
        self.speaker_ids = []
        for speaker_id in os.listdir(audio_dir):
            speaker_dir = os.path.join(audio_dir, speaker_id)
            if os.path.isdir(speaker_dir):
                for filename in os.listdir(speaker_dir):
                    if filename.endswith('.wav'):
                        self.audio_files.append(os.path.join(speaker_dir, filename))
                        self.speaker_ids.append(speaker_id)

        if lite is not None:
            self.audio_files = self.audio_files[:lite]
            self.speaker_ids = self.speaker_ids[:lite]

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_file = self.audio_files[idx]
        waveform, sample_rate = torchaudio.load(audio_file)
        return waveform, self.speaker_ids[idx]


def get_dataloader(split, batch_size=4, n_mels=128, lite=None):
    # Create MelSpectrogram transform
    transform = torchaudio.transforms.MelSpectrogram(n_mels=n_mels).to(torch.float32)

    split_audio_path = os.path.join(audio_path, split)

    # If lite is defined and the split is for validation, divide the lite number by 4
    if lite is not None and split == 'validation':
        lite = max(lite // 4, 1)

    dataset = AudioDataset(split_audio_path, lite)

    def collate_fn(examples):
        audios = []
        speaker_ids = []
        for example in examples:
            audio_tensor, speaker_id = example
            if audio_tensor.dim() > 1:
                audio_tensor = audio_tensor[0]
            encoded_speaker_id = encoder.transform([str(speaker_id)])[0]
            audios.append(audio_tensor)
            speaker_ids.append(encoded_speaker_id)

        # Convert audio list into tensor
        audios = torch.stack(audios)

        # Apply MelSpectrogram transform to audio
        audios = transform(audios).transpose(1, 2)  # shape: [batch_size, sequence_length, n_mels]

        speaker_ids = torch.tensor(speaker_ids)
        return {"audio_values": audios, "speaker_ids": speaker_ids}

    dataloader = DataLoader(dataset, collate_fn=collate_fn, shuffle=True, batch_size=batch_size)

    return dataloader, encoder
