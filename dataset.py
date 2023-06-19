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
from torchaudio.datasets import VoxCeleb1Identification  # noqa: E402

dataset_dict = {
    'train': ConcatDataset([load_dataset("edinburghcstr/ami", "ihm", split='train', cache_dir="data/ami"),
                            VoxCeleb1Identification(root='data/voxceleb1', subset='train', download=True)]),
    'validation': ConcatDataset([load_dataset("edinburghcstr/ami", "ihm", split='validation', cache_dir="data/ami"),
                                 VoxCeleb1Identification(root='data/voxceleb1', subset='dev', download=True)])
}

# File path to save/load the LabelEncoder
encoder_filepath = "data/encoder.pkl"


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
        all_speaker_ids.extend([str(example[2]) if isinstance(example, tuple) else example.get('speaker_id', 'NO_SPEAKER')
                                for example in tqdm(dataset_split, desc=f'Processing {split} dataset')])

    encoder.fit(all_speaker_ids)

    # Save the encoder to file
    with open(encoder_filepath, "wb") as f:
        pickle.dump(encoder, f)

    # Print the number of unique speaker_ids
    num_unique_speakers = len(set(all_speaker_ids))
    print(f"Number of unique speaker IDs: {num_unique_speakers}")


def get_dataloader(split, feature_type='melspectrogram', batch_size=4, n_mels=128,
                   max_duration=20, hop_duration=15, sample_rate=16000, lite=None):
    # Check feature type and create corresponding transform
    assert feature_type in ['melspectrogram', 'mfcc'], "Feature type must be either 'melspectrogram' or 'mfcc'"
    if feature_type == 'melspectrogram':
        transform = torchaudio.transforms.MelSpectrogram(n_mels=n_mels)
    else:
        transform = torchaudio.transforms.MFCC(n_mfcc=n_mels)  # assuming n_mels is reused as n_mfcc

    max_length_samples = int(max_duration * sample_rate)
    hop_length_samples = int(hop_duration * sample_rate)

    dataset = dataset_dict[split]

    if lite is not None:
        if split == "train":
            dataset = dataset.select(range(lite))
        elif split == "validation":
            dataset = dataset.select(range(lite // 4))

    def collate_fn(examples):
        audios = []
        speaker_ids = []
        for example in examples:
            audio_tensor, sample_rate, speaker_id = example if isinstance(example, tuple) else (example["audio"]["array"].float(), sample_rate, example.get('speaker_id', 'NO_SPEAKER'))
            audio_tensor = audio_tensor.float()
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
        audios = torch.stack(audios)

        # Apply transform (MelSpectrogram or MFCC) to audio
        audios = transform(audios)
        if feature_type == 'melspectrogram':
            audios = audios.transpose(1, 2)

        speaker_ids = torch.tensor(speaker_ids)
        return {"audio_values": audios, "speaker_ids": speaker_ids}

    dataset.set_format(type="torch", columns=["audio", "speaker_id"])
    dataloader = DataLoader(dataset, collate_fn=collate_fn, shuffle=True, batch_size=batch_size)

    return dataloader, encoder
