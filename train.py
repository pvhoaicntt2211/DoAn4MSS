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
    for mixture, vocals in loop:
        mixture = mixture.to(device)
        vocals = vocals.to(device)

        # Forward
        # Mô hình dự đoán spectrogram của vocal từ spectrogram của mixture
        predicted_vocals = model(mixture)

        # Tính loss
        # Lưu ý: Mô hình Sigmoid ở cuối trả về giá trị [0,1].
        # Ta cần nhân nó với mixture đầu vào để có được spectrogram vocal thực sự
        loss = loss_fn(predicted_vocals * mixture, vocals)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
        
    return total_loss / len(loader)

def validate_one_epoch(loader, model, loss_fn, device):
    model.eval() # Đặt model ở chế độ evaluation
    loop = tqdm(loader, leave=True)
    loop.set_description(f"Validating")
    
    total_loss = 0.0
    with torch.no_grad():
        for mixture, vocals in loop:
            mixture = mixture.to(device)
            vocals = vocals.to(device)

            predicted_vocals = model(mixture)
            loss = loss_fn(predicted_vocals * mixture, vocals)
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

    return total_loss / len(loader)


def main():
    parser = argparse.ArgumentParser(description="Train U-Net for vocal/accompaniment separation")
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
    model = UNet(in_channels=1, out_channels=1).to(config.DEVICE)
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

        train_loss = train_one_epoch(train_loader, model, optimizer, loss_fn, config.DEVICE)
        print(f"Average Training Loss: {train_loss:.4f}")

        valid_loss = validate_one_epoch(valid_loader, model, loss_fn, config.DEVICE)
        print(f"Average Validation Loss: {valid_loss:.4f}")

        # Lưu lại model nếu validation loss tốt hơn
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            checkpoint_path = os.path.join(args.checkpoint_dir, "best_model.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"✅ New best model saved to {checkpoint_path}")

if __name__ == "__main__":
    main()