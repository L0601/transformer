import argparse
import random

import torch

from data import (
    BOS,
    EOS,
    PAD,
    TOKEN_TO_ID,
    DEFAULT_TOTAL,
    decode_answer,
    encode_text,
    ensure_data,
    make_addition,
    read_rows,
    split_row,
    split_train_test,
)
from model import AdditionTransformer
from train import get_device


def load_model(path: str, device: torch.device) -> AdditionTransformer:
    model = AdditionTransformer(len(TOKEN_TO_ID), TOKEN_TO_ID[PAD]).to(device)
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


@torch.no_grad()
def predict(model, expr: str, device: torch.device, max_len: int = 5) -> str:
    src = torch.tensor([encode_text(expr)], dtype=torch.long, device=device)
    ids = [TOKEN_TO_ID[BOS]]
    for _ in range(max_len):
        tgt = torch.tensor([ids], dtype=torch.long, device=device)
        next_id = int(model(src, tgt)[0, -1].argmax().item())
        ids.append(next_id)
        if next_id == TOKEN_TO_ID[EOS]:
            break
    return decode_answer(ids[1:])


def evaluate_rows(model, rows: list[str], device: torch.device) -> float:
    correct = 0
    for row in rows:
        expr, answer = split_row(row)
        pred = predict(model, expr, device)
        correct += int(pred == answer)
    return correct / len(rows)


def random_case() -> tuple[str, str]:
    a = random.randint(0, 9999)
    b = random.randint(0, 9999 - a)
    expr, answer = split_row(make_addition(a, b))
    return expr, answer


def show_generalization(model, device: torch.device, count: int) -> None:
    print("随机泛化测试:")
    for _ in range(count):
        expr, answer = random_case()
        pred = predict(model, expr, device)
        print(f"{expr}{answer}, model={pred}, ok={pred == answer}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估 Transformer 加法模型")
    parser.add_argument("--data", default="data/additions.txt")
    parser.add_argument("--model", default="checkpoints/addition_transformer_v2.pt")
    parser.add_argument("--total", type=int, default=DEFAULT_TOTAL)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_data(args.data, total=args.total)
    rows = read_rows(args.data)
    _, test_rows = split_train_test(rows)
    if args.limit > 0:
        test_rows = test_rows[: args.limit]
    device = get_device()
    model = load_model(args.model, device)
    acc = evaluate_rows(model, test_rows, device)
    print(f"测试样本: {len(test_rows)}")
    print(f"完全匹配准确率: {acc:.4f}")
    show_generalization(model, device, args.samples)


if __name__ == "__main__":
    main()
