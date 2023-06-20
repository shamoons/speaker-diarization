# train.py
import torch
from tqdm import tqdm
import wandb
from dataset import get_dataloader
from utils import get_arg_parser, save_checkpoint, load_checkpoint
from model import SpeakerIdentificationModel

torch.manual_seed(0)


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


def main():
    parser = get_arg_parser()
    args = parser.parse_args()
    # Set device
    device = "cpu"
    if args.use_mps and torch.backends.mps.is_available():
        device = "mps"
    elif args.use_cuda and torch.cuda.is_available():
        device = "cuda"

    print(f"Using device: {device}")
    device = torch.device(device)

    # Initialize wandb
    wandb_run = wandb.init(project="speech-inpainting", config=args.__dict__)
    print("wandb dir:", wandb.run.dir)

    train_dataloader, encoder = get_dataloader("train", batch_size=args.batch_size,
                                               n_mels=args.d_model, lite=args.lite, feature_type=args.feature_type)
    num_classes = len(encoder.classes_)

    model = SpeakerIdentificationModel(args.d_model, args.nhead, args.num_layers, args.dim_feedforward, num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.checkpoint_path:
        load_checkpoint(args.checkpoint_path, model, optimizer)

    val_dataloader, _ = get_dataloader("validation", batch_size=args.batch_size,
                                       n_mels=args.d_model, lite=args.lite, feature_type=args.feature_type)

    for epoch in range(args.epochs):
        print(f'Epoch {epoch+1}')
        train_loss, train_accuracy = train_epoch(model, train_dataloader, optimizer, criterion, device)
        val_loss, val_accuracy = validate_epoch(model, val_dataloader, criterion, device)
        print(f'Train Loss: {train_loss:.3f}\t'
              f'Train Accuracy: {train_accuracy:.3f}\t'
              f'Validation Loss: {val_loss:.3f}\t'
              f'Validation Accuracy: {val_accuracy:.3f}\n')
        wandb_run.log({"train_loss": train_loss, "val_loss": val_loss, "train_accuracy": train_accuracy, "val_accuracy": val_accuracy})

        save_checkpoint({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, 'checkpoint.pth')

    # Finish the wandb run
    wandb.finish()


if __name__ == "__main__":
    main()
