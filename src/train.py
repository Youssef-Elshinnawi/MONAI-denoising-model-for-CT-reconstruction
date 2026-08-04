"""Training loop for ResidualDenoiser: MSE loss, Adam, checkpoint on best val loss."""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import build_dataset, TRAIN_PATIENTS, VAL_PATIENTS
from model import ResidualDenoiser


def train_one_epoch(model, dataloader, optimizer, loss_fn, device) -> float:
    model.train()
    running_loss = 0.0

    for batch in dataloader:
        low = batch["low"].to(device)
        full = batch["full"].to(device)

        output = model(low)
        loss = loss_fn(output, full)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * low.shape[0]

    return running_loss / len(dataloader.dataset)


def validate_one_epoch(model, dataloader, loss_fn, device) -> float:
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            low = batch["low"].to(device)
            full = batch["full"].to(device)

            output = model(low)
            loss = loss_fn(output, full)

            running_loss += loss.item() * low.shape[0]

    return running_loss / len(dataloader.dataset)


def train(model, train_ds, val_ds, num_epochs: int, checkpoint_dir: str, device):
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_path = os.path.join(checkpoint_dir, "training_log.csv")

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=2)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")

    # Flushed every epoch so the curve survives a Colab disconnect mid-run.
    with open(log_path, "w") as log_file:
        log_file.write("epoch,train_loss,val_loss\n")

        for epoch in range(num_epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
            val_loss = validate_one_epoch(model, val_loader, loss_fn, device)

            print(f"epoch {epoch+1}/{num_epochs}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")
            log_file.write(f"{epoch+1},{train_loss:.6f},{val_loss:.6f}\n")
            log_file.flush()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model.pt"))


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"training on: {device}")

    DATA_ROOT = "/content/drive/MyDrive/MONAI-denoising/data/raw"
    CHECKPOINT_DIR = "/content/drive/MyDrive/MONAI-denoising/checkpoints"

    train_ds = build_dataset(TRAIN_PATIENTS, DATA_ROOT)
    val_ds = build_dataset(VAL_PATIENTS, DATA_ROOT)

    model = ResidualDenoiser().to(device)

    train(model, train_ds, val_ds, num_epochs=100, checkpoint_dir=CHECKPOINT_DIR, device=device)
