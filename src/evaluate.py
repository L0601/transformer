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
    pad_items,
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


@torch.inference_mode()
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


@torch.inference_mode()
def predict_batch(
    model,
    exprs: list[str],
    device: torch.device,
    max_len: int = 5,
) -> list[str]:
    if not exprs:
        return []

    samples = [
        {"src": torch.tensor(encode_text(expr), dtype=torch.long)}
        for expr in exprs
    ]
    src = pad_items(samples, "src").to(device)
    ids = torch.full(
        (len(exprs), 1),
        TOKEN_TO_ID[BOS],
        dtype=torch.long,
        device=device,
    )
    finished = torch.zeros(len(exprs), dtype=torch.bool, device=device)

    for _ in range(max_len):
        next_ids = model(src, ids)[:, -1].argmax(dim=-1)
        next_ids = torch.where(
            finished,
            torch.full_like(next_ids, TOKEN_TO_ID[EOS]),
            next_ids,
        )
        ids = torch.cat([ids, next_ids.unsqueeze(1)], dim=1)
        finished |= next_ids.eq(TOKEN_TO_ID[EOS])
        if bool(finished.all()):
            break

    return [decode_answer(row.tolist()) for row in ids[:, 1:].cpu()]


def evaluate_rows(model, rows: list[str], device: torch.device) -> float:
    correct = 0
    for row in rows:
        expr, answer = split_row(row)
        pred = predict(model, expr, device)
        correct += int(pred == answer)
    return correct / len(rows)


def evaluate_rows_batch(
    model,
    rows: list[str],
    device: torch.device,
    batch_size: int,
) -> float:
    correct = 0
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        pairs = [split_row(row) for row in batch_rows]
        exprs = [expr for expr, _ in pairs]
        answers = [answer for _, answer in pairs]
        preds = predict_batch(model, exprs, device)
        correct += sum(int(pred == answer) for pred, answer in zip(preds, answers))
    return correct / len(rows)


def random_case() -> tuple[str, str]:
    a = random.randint(0, 9999)
    b = random.randint(0, 9999 - a)
    expr, answer = split_row(make_addition(a, b))
    return expr, answer


# 训练分布外的变长样例：训练时全是 4 位补零、左右等宽、答案 ≤ 4 位
# 这里用来观察模型对不同类型 OOD 输入的退化模式
VARIED_LENGTH_CASES: list[tuple[str, str, str]] = [
    ("不补零短数", "1+2=", "3"),
    ("不补零短数", "12+5=", "17"),
    ("不补零短数", "100+200=", "300"),
    ("位数不一致", "1+999=", "1000"),
    ("位数不一致", "9+1=", "10"),
    ("位数不一致", "5+1234=", "1239"),
    ("答案溢出 4 位", "9999+1=", "10000"),
    ("答案溢出 4 位", "9999+9999=", "19998"),
    ("输入超训练范围", "12345+1=", "12346"),
    ("输入超训练范围", "10000+10000=", "20000"),
]


def show_generalization(model, device: torch.device, count: int) -> None:
    print("随机泛化测试（训练同分布）:")
    for _ in range(count):
        expr, answer = random_case()
        pred = predict(model, expr, device)
        print(f"  {expr}{answer}, model={pred}, ok={pred == answer}")

    print("\n变长泛化测试（训练分布外）:")
    last_category = None
    for category, expr, answer in VARIED_LENGTH_CASES:
        if category != last_category:
            print(f"  [{category}]")
            last_category = category
        # 答案长度可能超过默认 5，留一点 EOS 余量
        pred = predict(model, expr, device, max_len=max(len(answer) + 2, 6))
        print(f"  {expr}{answer}, model={pred}, ok={pred == answer}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估 Transformer 加法模型")
    parser.add_argument("--data", default="data/additions.txt")
    parser.add_argument("--model", default="checkpoints/addition_transformer_v2.pt")
    parser.add_argument("--total", type=int, default=DEFAULT_TOTAL)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=512)
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
    acc = evaluate_rows_batch(model, test_rows, device, args.batch_size)
    print(f"测试样本: {len(test_rows)}")
    print(f"完全匹配准确率: {acc:.4f}")
    show_generalization(model, device, args.samples)


if __name__ == "__main__":
    main()
