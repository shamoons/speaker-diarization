# dataset.py
import torch
from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
import torchaudio
from tqdm import tqdm

# Load all splits of the dataset
dataset_dict = load_dataset("edinburghcstr/ami", "ihm", cache_dir="data/ami")

# Create and fit a LabelEncoder to all the speaker_ids in all splits of the dataset
encoder = LabelEncoder()
all_speaker_ids = []
print("Starting to fit the LabelEncoder...")
for split in dataset_dict.keys():
    # use 'NO_SPEAKER' if no speaker_id is provided
    dataset_split = dataset_dict[split]
    all_speaker_ids.extend([example.get('speaker_id', 'NO_SPEAKER') for example in tqdm(dataset_split, desc=f'Processing {split} dataset')])

encoder.fit(all_speaker_ids)

# Print the number of unique speaker_ids
num_unique_speakers = len(set(all_speaker_ids))
print(f"Number of unique speaker IDs: {num_unique_speakers}")


def get_dataloader(split, batch_size=4, n_mels=128, max_duration=20, hop_duration=19, sample_rate=16000):
    # Create a MelSpectrogram transformation with the given number of Mel frequencies (n_mels)
    mel_transform = torchaudio.transforms.MelSpectrogram(n_mels=n_mels)

    max_length_samples = int(max_duration * sample_rate)
    hop_length_samples = int(hop_duration * sample_rate)

    dataset = dataset_dict[split]

    def collate_fn(examples):
        audios = []
        speaker_ids = []
        for example in examples:
            audio_tensor = example["audio"]["array"].float()
            speaker_id = example.get('speaker_id', 'NO_SPEAKER')  # use 'NO_SPEAKER' if no speaker_id is provided
            encoded_speaker_id = encoder.transform([speaker_id])[0]

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

        # Convert padded audio to Mel spectrograms
        audios = mel_transform(audios).transpose(1, 2)

        speaker_ids = torch.tensor(speaker_ids)
        return {"audio_values": audios, "speaker_ids": speaker_ids}

    dataset.set_format(type="torch", columns=["audio", "speaker_id"])
    dataloader = DataLoader(dataset, collate_fn=collate_fn, shuffle=True, batch_size=batch_size)

    return dataloader, encoder
