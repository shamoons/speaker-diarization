# prepare_data.py
import torch
import os
import torchaudio
from datasets import load_dataset
from torchaudio.datasets import LIBRISPEECH, VoxCeleb1Identification
from tqdm import tqdm

dataset_dict = {
    'train': [
        load_dataset("edinburghcstr/ami", "ihm", split='train', cache_dir="data/ami"),
        VoxCeleb1Identification(root='data/voxceleb1', subset='train', download=True),
        LIBRISPEECH(root='data/LIBRISPEECH', url='train-clean-100', folder_in_archive='LibriSpeech', download=True)
    ],
    'validation': [
        load_dataset("edinburghcstr/ami", "ihm", split='validation', cache_dir="data/ami"),
        VoxCeleb1Identification(root='data/voxceleb1', subset='dev', download=True),
        LIBRISPEECH(root='data/LIBRISPEECH', url='dev-clean', folder_in_archive='LibriSpeech', download=True)
    ]
}


def save_audio(audio, sample_rate, speaker_id, audio_id, split):
    path = f"data/audio/{split}/{speaker_id}"
    os.makedirs(path, exist_ok=True)
    audio = audio.unsqueeze(0)
    audio = audio.to(torch.float32)
    torchaudio.save(f"{path}/{audio_id}.wav", audio, sample_rate)


def get_speaker_ids(example):
    if isinstance(example, tuple):
        if len(example) == 4:
            return str(f"voxceleb1-{example[2]}")
        elif len(example) == 6:
            return str(f"librispeech-{example[3]}")
    else:
        return example.get('speaker_id', 'NO_SPEAKER')


def process_and_save_audio_data():
    max_duration = 20  # seconds
    hop_duration = 15  # seconds
    sample_rate = 16000  # Hz
    max_length_samples = int(max_duration * sample_rate)
    hop_length_samples = int(hop_duration * sample_rate)

    for split, datasets in dataset_dict.items():
        print(f"Processing {split} datasets...")
        for dataset in datasets:
            print(f"Processing {type(dataset).__name__}...")
            for i, example in tqdm(enumerate(dataset), desc=f'Processing {split} dataset', total=len(dataset)):
                if isinstance(example, tuple):
                    audio_tensor, sample_rate = example[0], example[1]
                    speaker_id = get_speaker_ids(example)
                elif isinstance(example, dict):
                    audio_tensor, sample_rate = torch.from_numpy(example['audio']['array']), example['audio']['sampling_rate']
                    speaker_id = example['speaker_id']
                else:
                    raise ValueError(f'Unexpected data type in example: {type(example)}. Expected tuple or dict.')

                if len(audio_tensor.shape) > 1:
                    audio_tensor = audio_tensor.flatten()

                for start in range(0, max(1, len(audio_tensor) - max_length_samples + 1), hop_length_samples):
                    end = start + max_length_samples
                    segment = audio_tensor[start:end]
                    if len(segment) < max_length_samples:
                        segment = torch.Tensor(segment)
                        segment = torch.nn.functional.pad(segment, (0, max_length_samples - len(segment)))
                    save_audio(segment, sample_rate, speaker_id, f"{i}_{start // hop_length_samples}", split)


if __name__ == "__main__":
    process_and_save_audio_data()
