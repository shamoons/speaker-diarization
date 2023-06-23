# dataset.py
import os
import pickle

import torch
from torch.utils.data import DataLoader, ConcatDataset
from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder
import torchaudio
from tqdm import tqdm

# Load all splits of the dataset
os.makedirs('data/voxceleb1', exist_ok=True)
from torchaudio.datasets import VoxCeleb1Identification  # noqa E402

dataset_dict = {
    'train': ConcatDataset([
        load_dataset("edinburghcstr/ami", "ihm", split='train', cache_dir="data/ami"),
        VoxCeleb1Identification(root='data/voxceleb1', subset='train', download=True)
    ]),
    'validation': ConcatDataset([
        load_dataset("edinburghcstr/ami", "ihm", split='validation', cache_dir="data/ami"),
        VoxCeleb1Identification(root='data/voxceleb1', subset='dev', download=True)
    ])
}

# File path to save/load the LabelEncoder
encoder_filepath = "data/encoder.pkl"


def get_speaker_ids(example):
    if isinstance(example, tuple):
        return str(f"voxceleb1-{example[2]}")
    else:
        return example.get('speaker_id', 'NO_SPEAKER')


# Check if the encoder file exists
if os.path.isfile(encoder_filepath):
    # Load the encoder from file
    with open(encoder_filepath, "rb") as f:
        encoder = pickle.load(f)
else:
    # Create and fit a LabelEncoder to all the speaker_ids in all splits of the dataset
    encoder = LabelEncoder()
    all_speaker_ids = []
    print("Starting to fit the LabelEncoder...")
    for split in dataset_dict.keys():
        # use 'NO_SPEAKER' if no speaker_id is provided
        dataset_split = dataset_dict[split]
        for example in tqdm(dataset_split, desc=f'Processing {split} dataset'):
            speaker_id = get_speaker_ids(example)
            all_speaker_ids.append(speaker_id)

    encoder.fit(all_speaker_ids)

    # Save the encoder to file
    with open(encoder_filepath, "wb") as f:
        pickle.dump(encoder, f)

    # Print the number of unique speaker_ids
    num_unique_speakers = len(set(all_speaker_ids))
    print(f"Number of unique speaker IDs: {num_unique_speakers}")


class CustomSubset(torch.utils.data.Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)

    def __getitem__(self, idx):
        data = self.dataset[self.indices[idx]]
        if isinstance(data, dict):  # if data is a dictionary, use the key
            return data['audio']['array'], data['speaker_id']
        elif isinstance(data, tuple):  # if data is a tuple, use the indices
            return data[0].flatten(), str(f"voxceleb1-{data[2]}")
        else:
            raise ValueError('Data type not recognized. It should be either a dictionary or a tuple.')


def get_dataloader(split, batch_size=4, n_mels=128,
                   max_duration=20, hop_duration=15, sample_rate=16000, lite=None):
    # Create MelSpectrogram transform
    transform = torchaudio.transforms.MelSpectrogram(n_mels=n_mels).to(torch.float32)

    max_length_samples = int(max_duration * sample_rate)
    hop_length_samples = int(hop_duration * sample_rate)

    dataset = dataset_dict[split]

    if lite is not None:
        if split == "train":
            indices = list(range(lite))
            dataset = CustomSubset(dataset, indices)
        elif split == "validation":
            indices = list(range(lite // 4))
            dataset = CustomSubset(dataset, indices)

    def collate_fn(examples):
        audios = []
        speaker_ids = []
        for example in examples:
            if len(example) == 2:
                audio_tensor, speaker_id = example
                if not torch.is_tensor(audio_tensor):
                    audio_tensor = torch.from_numpy(audio_tensor)

            elif len(example) == 4:
                audio_tensor, _, speaker_id, _ = example
                speaker_id = str(f"voxceleb1-{speaker_id}")
                audio_tensor = audio_tensor.flatten()
            elif isinstance(example, dict):
                audio_tensor = example['audio']['array']
                # Turn audio_tensor to torch tensor from numpy array
                audio_tensor = torch.from_numpy(audio_tensor)
                speaker_id = example['speaker_id']
            else:
                raise ValueError(f'Unexpected data type in example: {type(example)}. Expected tuple or dict.')

            encoded_speaker_id = encoder.transform([str(speaker_id)])[0]

            for start in range(0, max(1, len(audio_tensor) - max_length_samples + 1), hop_length_samples):
                end = start + max_length_samples
                segment = audio_tensor[start:end]
                # If the audio segment is shorter than max_duration, pad it
                if len(segment) < max_length_samples:
                    segment = torch.nn.functional.pad(segment, (0, max_length_samples - len(segment)))
                audios.append(segment)
                speaker_ids.append(encoded_speaker_id)  # Duplicate speaker_id for each segment

        # Convert audio list into tensor
        audios = [audio.float() for audio in audios]
        audios = torch.stack(audios)

        # Apply MelSpectrogram transform to audio
        audios = transform(audios).transpose(1, 2)  # shape: [batch_size, sequence_length, n_mels]

        speaker_ids = torch.tensor(speaker_ids)
        return {"audio_values": audios, "speaker_ids": speaker_ids}

    dataloader = DataLoader(dataset, collate_fn=collate_fn, shuffle=True, batch_size=batch_size)

    return dataloader, encoder
