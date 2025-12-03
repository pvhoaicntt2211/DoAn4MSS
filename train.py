# file: train.py

import argparse
import os

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from src.dataset import MUSDBDataset
from src.model import UNet

def train_one_epoch(loader, model, optimizer, loss_fn, device):
    model.train() # Đặt model ở chế độ training
    loop = tqdm(loader, leave=True)
    loop.set_description(f"Training")
    
    total_loss = 0.0
    stem_losses = {stem: 0.0 for stem in config.STEMS}
    
    for mixture, stems_target in loop:
        # mixture: (Batch, 1 channel, Frequency bins, Time frames)
        mixture = mixture.to(device)  # (B, 1, F, T)
        # stems_target: (Batch, 4 stems, Frequency bins, Time frames)
        stems_target = stems_target.to(device)  # (B, 4, F, T)

        # Forward
        # Model predicts 4 masks (one for each stem)
        predicted_masks = model(mixture)  # (B, 4, F, T)

        # Apply masks to mixture to get predicted stems
        # Expand mixture to match stems shape
        mixture_expanded = mixture.expand(-1, 4, -1, -1)  # (B, 4, F, T)
        predicted_stems = predicted_masks * mixture_expanded

        # Calculate weighted loss for each stem
        loss = 0.0
        for i, stem in enumerate(config.STEMS):
            stem_loss = loss_fn(predicted_stems[:, i:i+1, :, :], stems_target[:, i:i+1, :, :])
            weighted_loss = config.STEM_WEIGHTS[stem] * stem_loss
            loss += weighted_loss
            stem_losses[stem] += stem_loss.item()

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
        
    # Calculate average losses
    avg_total_loss = total_loss / len(loader)
    avg_stem_losses = {stem: stem_losses[stem] / len(loader) for stem in config.STEMS}
    
    return avg_total_loss, avg_stem_losses

def validate_one_epoch(loader, model, loss_fn, device):
    model.eval() # Đặt model ở chế độ evaluation
    loop = tqdm(loader, leave=True)
    loop.set_description(f"Validating")
    
    total_loss = 0.0
    stem_losses = {stem: 0.0 for stem in config.STEMS}
    
    with torch.no_grad():
        for mixture, stems_target in loop:
            # mixture: (Batch, 1 channel, Frequency bins, Time frames)
            mixture = mixture.to(device)  # (B, 1, F, T)
            # stems_target: (Batch, 4 stems, Frequency bins, Time frames)
            stems_target = stems_target.to(device)  # (B, 4, F, T)

            # Forward
            predicted_masks = model(mixture)  # (B, 4, F, T)
            
            # Apply masks to mixture
            mixture_expanded = mixture.expand(-1, 4, -1, -1)
            predicted_stems = predicted_masks * mixture_expanded

            # Calculate weighted loss for each stem
            loss = 0.0
            for i, stem in enumerate(config.STEMS):
                stem_loss = loss_fn(predicted_stems[:, i:i+1, :, :], stems_target[:, i:i+1, :, :])
                weighted_loss = config.STEM_WEIGHTS[stem] * stem_loss
                loss += weighted_loss
                stem_losses[stem] += stem_loss.item()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

    avg_total_loss = total_loss / len(loader)
    avg_stem_losses = {stem: stem_losses[stem] / len(loader) for stem in config.STEMS}
    
    return avg_total_loss, avg_stem_losses


def main():
    parser = argparse.ArgumentParser(description="Train U-Net for 4-stem music source separation")
    parser.add_argument("--train-dir", default=config.TRAIN_DIR, help="Thư mục dữ liệu train")
    parser.add_argument("--valid-dir", default=config.VALID_DIR, help="Thư mục dữ liệu valid")
    parser.add_argument("--checkpoint-dir", default=config.CHECKPOINT_DIR, help="Thư mục lưu checkpoint")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS, help="Số epoch")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS, help="Số worker cho DataLoader")
    args = parser.parse_args()

    # Tạo thư mục checkpoint nếu chưa có
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Khởi tạo model, loss, optimizer
    model = UNet(in_channels=1, out_channels=4).to(config.DEVICE)  # 4 output channels for 4 stems
    loss_fn = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Chuẩn bị Dataloaders
    train_dataset = MUSDBDataset(root_dir=args.train_dir)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True
    )

    valid_dataset = MUSDBDataset(root_dir=args.valid_dir)
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False
    )

    best_valid_loss = float('inf')

    # Vòng lặp huấn luyện chính
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")

        train_loss, train_stem_losses = train_one_epoch(train_loader, model, optimizer, loss_fn, config.DEVICE)
        print(f"Average Training Loss: {train_loss:.4f}")
        for stem, loss_val in train_stem_losses.items():
            print(f"  {stem}: {loss_val:.4f}")

        valid_loss, valid_stem_losses = validate_one_epoch(valid_loader, model, loss_fn, config.DEVICE)
        print(f"Average Validation Loss: {valid_loss:.4f}")
        for stem, loss_val in valid_stem_losses.items():
            print(f"  {stem}: {loss_val:.4f}")

        # Lưu lại model nếu validation loss tốt hơn
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            checkpoint_path = os.path.join(args.checkpoint_dir, "best_model.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"✅ New best model saved to {checkpoint_path}")

if __name__ == "__main__":
    main()