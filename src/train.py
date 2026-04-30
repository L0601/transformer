import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from data import AdditionDataset, PAD, TOKEN_TO_ID, ensure_data, pad_batch, read_rows
from model import AdditionTransformer


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_loader(rows: list[str], batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        AdditionDataset(rows),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=pad_batch,
    )


def train_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        src = batch["src"].to(device)
        tgt_in = batch["tgt_in"].to(device)
        tgt_out = batch["tgt_out"].to(device)
        optimizer.zero_grad()
        logits = model(src, tgt_in)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
        loss.backward()
        optimizer.step()
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
    parser.add_argument("--data", default="data/additions.txt")
    parser.add_argument("--model", default="checkpoints/addition_transformer.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_data(args.data)
    rows = read_rows(args.data)
    train_rows = rows[:6000]
    device = get_device()
    loader = build_loader(train_rows, args.batch_size, shuffle=True)
    model = AdditionTransformer(len(TOKEN_TO_ID), TOKEN_TO_ID[PAD]).to(device)
    loaded = load_checkpoint_if_exists(model, args.model, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=TOKEN_TO_ID[PAD])

    print(f"使用设备: {device}")
    print(f"训练样本: {len(train_rows)}")
    if loaded:
        print(f"已加载已有模型参数: {args.model}")
    else:
        print("未找到已有模型，从头训练")
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, loader, optimizer, criterion, device)
        print(f"epoch={epoch:03d}, loss={loss:.4f}")
    save_checkpoint(model, args.model)
    print(f"模型已保存: {args.model}")


if __name__ == "__main__":
    main()
