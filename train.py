# train.py
import torch
import time
import wandb
from torch.optim.lr_scheduler import LambdaLR
from dataset import get_dataloader
from utils import get_arg_parser, save_checkpoint, load_checkpoint
from model import SpeakerIdentificationModel
from epoch import train_epoch, validate_epoch


torch.manual_seed(0)


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
    wandb_run = wandb.init(project="speaker-diarization", config=args.__dict__)
    print("wandb dir:", wandb.run.dir)

    train_dataloader, encoder = get_dataloader("train", batch_size=args.batch_size,
                                               n_mels=args.d_model, lite=args.lite)
    num_classes = len(encoder.classes_)

    model = SpeakerIdentificationModel(args.d_model, args.nhead, args.num_layers,
                                       args.dim_feedforward, num_classes, n_time_steps=args.n_time_steps,
                                       dropout=args.dropout).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    warmup_steps = int(args.epochs * args.warmup_steps)  # % of total steps
    print("Warmup steps:", warmup_steps)

    def lr_lambda(step):

        lr = args.lr * min((step + 1)
                           ** -0.5, (step + 1) * warmup_steps ** -1.5)
        return lr

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    if args.checkpoint_path:
        load_checkpoint(args.checkpoint_path, model, optimizer)

    val_dataloader, _ = get_dataloader("validation", batch_size=args.batch_size,
                                       n_mels=args.d_model, lite=args.lite)

    for epoch in range(args.epochs):
        start_time = time.time()
        print(f'Epoch {epoch+1}')
        train_loss, train_accuracy = train_epoch(model, train_dataloader, optimizer, criterion, device)
        val_loss, val_accuracy = validate_epoch(model, val_dataloader, criterion, device)

        # Adjust learning rate
        scheduler.step()

        end_time = time.time()
        epoch_time = end_time - start_time

        print(f'Train Loss: {train_loss:.3f}\t'
              f'Train Accuracy: {train_accuracy:.3f}\t'
              f'Validation Loss: {val_loss:.3f}\t'
              f'Validation Accuracy: {val_accuracy:.3f}\t'
              f'Learning Rate: {scheduler.get_last_lr()[0]:.4e}\t'
              f'Epoch Time: {epoch_time:.2f} seconds\n')

        wandb_run.log({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
            "learning_rate": scheduler.get_last_lr()[0]
        })

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
