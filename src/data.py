import random
from pathlib import Path


PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
TOKENS = [PAD, BOS, EOS, "+", "="] + list("0123456789")
TOKEN_TO_ID = {token: idx for idx, token in enumerate(TOKENS)}
ID_TO_TOKEN = {idx: token for token, idx in TOKEN_TO_ID.items()}
DEFAULT_TOTAL = 50000
TRAIN_RATIO = 0.8
# 单个加数最多多少位；答案最多 MAX_DIGITS+1 位
MAX_DIGITS = 4


def make_addition(a: int, b: int) -> str:
    # 不补零，让位数自然变化（v3 数据格式）
    return f"{a}+{b}={a + b}"


def digit_capacity(digits: int) -> int:
    # 1 位含 0~9 共 10 个；其它位排除前导零
    if digits == 1:
        return 10
    return 9 * 10 ** (digits - 1)


def sample_in_digit(rng: random.Random, digits: int) -> int:
    if digits == 1:
        return rng.randint(0, 9)
    return rng.randint(10 ** (digits - 1), 10 ** digits - 1)


def fill_bucket(
    rng: random.Random,
    da: int,
    db: int,
    count: int,
    seen: set[str],
) -> None:
    # 受限于桶容量，最多取 capacity 条不重复样本
    capacity = digit_capacity(da) * digit_capacity(db)
    take = min(count, capacity)
    start = len(seen)
    while len(seen) - start < take:
        a = sample_in_digit(rng, da)
        b = sample_in_digit(rng, db)
        seen.add(make_addition(a, b))


def generate_data(path: str, total: int = DEFAULT_TOTAL, seed: int = 42) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    # 按 (a 位数, b 位数) 16 个桶均匀采样，让训练分布覆盖所有位数组合
    per_bucket = total // (MAX_DIGITS * MAX_DIGITS)
    seen: set[str] = set()
    for da in range(1, MAX_DIGITS + 1):
        for db in range(1, MAX_DIGITS + 1):
            fill_bucket(rng, da, db, per_bucket, seen)
    rows = list(seen)
    rng.shuffle(rows)
    target.write_text("\n".join(rows) + "\n", encoding="utf-8")


def ensure_data(path: str, total: int = DEFAULT_TOTAL) -> None:
    # v3 起每桶受容量限制，实际行数会小于 total，因此只判断文件是否存在
    data_path = Path(path)
    if not data_path.exists():
        generate_data(path, total=total)


def read_rows(path: str) -> list[str]:
    return Path(path).read_text(encoding="utf-8").splitlines()


def split_row(row: str) -> tuple[str, str]:
    expr, answer = row.split("=")
    return expr + "=", answer


def split_train_test(rows: list[str]) -> tuple[list[str], list[str]]:
    train_size = int(len(rows) * TRAIN_RATIO)
    return rows[:train_size], rows[train_size:]


def encode_text(text: str) -> list[int]:
    return [TOKEN_TO_ID[ch] for ch in text]


def decode_ids(ids: list[int]) -> str:
    chars = []
    for idx in ids:
        token = ID_TO_TOKEN[int(idx)]
        if token == EOS:
            break
        if token not in (PAD, BOS):
            chars.append(token)
    return "".join(chars)


def encode_answer(answer: str) -> list[int]:
    return encode_text(answer[::-1])


def decode_answer(ids: list[int]) -> str:
    return decode_ids(ids)[::-1]


class AdditionDataset:
    def __init__(self, rows: list[str]):
        self.samples = [split_row(row) for row in rows]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        import torch

        expr, answer = self.samples[idx]
        src = encode_text(expr)
        answer_ids = encode_answer(answer)
        tgt_in = [TOKEN_TO_ID[BOS]] + answer_ids
        tgt_out = answer_ids + [TOKEN_TO_ID[EOS]]
        return {
            "src": torch.tensor(src, dtype=torch.long),
            "tgt_in": torch.tensor(tgt_in, dtype=torch.long),
            "tgt_out": torch.tensor(tgt_out, dtype=torch.long),
        }


def pad_batch(batch: list[dict]):
    return {
        "src": pad_items(batch, "src"),
        "tgt_in": pad_items(batch, "tgt_in"),
        "tgt_out": pad_items(batch, "tgt_out"),
    }


def pad_items(batch: list[dict], key: str):
    import torch

    items = [item[key] for item in batch]
    return torch.nn.utils.rnn.pad_sequence(
        items,
        batch_first=True,
        padding_value=TOKEN_TO_ID[PAD],
    )
