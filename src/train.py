import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from data import (
    AdditionDataset,
    DEFAULT_TOTAL,
    PAD,
    TOKEN_TO_ID,
    ensure_data,
    pad_batch,
    read_rows,
    split_train_test,
)
from model import AdditionTransformer


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    return torch.device("cpu")


def build_loader(
    rows: list[str],
    batch_size: int,
    shuffle: bool,
    pin_memory: bool,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        AdditionDataset(rows),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=pad_batch,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )


def train_epoch(model, loader, optimizer, criterion, device, scaler, use_amp) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        src = batch["src"].to(device, non_blocking=True)
        tgt_in = batch["tgt_in"].to(device, non_blocking=True)
        tgt_out = batch["tgt_out"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(src, tgt_in)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.item())
    return total_loss / len(loader)


def save_checkpoint(model, path: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict()}, target)


def load_checkpoint_if_exists(model, path: str, device: torch.device) -> bool:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        return False
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 Transformer 加法模型")
    parser.add_argument("--data", default="data/additions_v3.txt")
    parser.add_argument("--model", default="checkpoints/addition_transformer_v4.pt")
    parser.add_argument("--total", type=int, default=DEFAULT_TOTAL)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_data(args.data, total=args.total)
    rows = read_rows(args.data)
    train_rows, _ = split_train_test(rows)
    device = get_device()
    if device.type == "cuda":
        torch.backends.cuda.matmul.fp32_precision = "tf32"
    use_amp = device.type == "cuda" and not args.no_amp
    loader = build_loader(
        train_rows,
        args.batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
        num_workers=args.num_workers,
    )
    model = AdditionTransformer(len(TOKEN_TO_ID), TOKEN_TO_ID[PAD]).to(device)
    loaded = load_checkpoint_if_exists(model, args.model, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5,
    )
    criterion = nn.CrossEntropyLoss(ignore_index=TOKEN_TO_ID[PAD])
    scaler_device = "cuda" if device.type == "cuda" else "cpu"
    scaler = torch.amp.GradScaler(scaler_device, enabled=use_amp)

    print(f"使用设备: {device}")
    print(f"训练样本: {len(train_rows)}")
    print(f"AMP: {use_amp}")
    print(f"batch_size: {args.batch_size}")
    if loaded:
        print(f"已加载已有模型参数: {args.model}")
    else:
        print("未找到已有模型，从头训练")
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, loader, optimizer, criterion, device, scaler, use_amp)
        scheduler.step(loss)
        lr = optimizer.param_groups[0]["lr"]
        print(f"epoch={epoch:03d}, loss={loss:.4f}, lr={lr:.6f}")
    save_checkpoint(model, args.model)
    print(f"模型已保存: {args.model}")


if __name__ == "__main__":
    main()
