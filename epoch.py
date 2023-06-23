# epoch.py
import torch
from tqdm import tqdm


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct_predictions = 0.0
    total_predictions = 0.0
    progress_bar = tqdm(dataloader, desc="Train", dynamic_ncols=True)
    for batch in progress_bar:
        audio_values = batch["audio_values"].to(device)
        speaker_ids = batch["speaker_ids"].to(device)
        optimizer.zero_grad()

        # Your model forward pass
        outputs = model(audio_values)

        # compute loss
        loss = criterion(outputs, speaker_ids)

        loss.backward()

        optimizer.step()
        running_loss += loss.item() * audio_values.size(0)

        # Compute accuracy
        _, predicted = torch.max(outputs, 1)
        total_predictions += speaker_ids.size(0)
        correct_predictions += (predicted == speaker_ids).sum().item()

        progress_bar.set_postfix({'loss': '{:.3f}'.format(loss.item() / len(batch))})

    epoch_loss = running_loss / len(dataloader.dataset)
    accuracy = correct_predictions / total_predictions
    return epoch_loss, accuracy


def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0.0
    total_predictions = 0.0
    progress_bar = tqdm(dataloader, desc="Validate", dynamic_ncols=True)
    with torch.no_grad():
        for batch in progress_bar:
            audio_values = batch["audio_values"].to(device)
            speaker_ids = batch["speaker_ids"].to(device)
            # Your model forward pass
            outputs = model(audio_values)
            # compute loss
            loss = criterion(outputs, speaker_ids)
            running_loss += loss.item() * audio_values.size(0)

            # Compute accuracy
            _, predicted = torch.max(outputs, 1)
            total_predictions += speaker_ids.size(0)
            correct_predictions += (predicted == speaker_ids).sum().item()

            progress_bar.set_postfix({'loss': '{:.3f}'.format(loss.item() / len(batch))})

    epoch_loss = running_loss / len(dataloader.dataset)
    accuracy = correct_predictions / total_predictions
    return epoch_loss, accuracy
