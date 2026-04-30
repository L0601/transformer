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
MAX_NUMBER = 9999
NUMBER_WIDTH = 4
ANSWER_WIDTH = 4


def make_addition(a: int, b: int) -> str:
    return f"{a:0{NUMBER_WIDTH}d}+{b:0{NUMBER_WIDTH}d}={a + b:0{ANSWER_WIDTH}d}"


def generate_data(path: str, total: int = DEFAULT_TOTAL, seed: int = 42) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    rows = []
    seen = set()
    while len(rows) < total:
        a = random.randint(0, MAX_NUMBER)
        b = random.randint(0, MAX_NUMBER - a)
        row = make_addition(a, b)
        if row not in seen:
            seen.add(row)
            rows.append(row)
    target.write_text("\n".join(rows) + "\n", encoding="utf-8")


def ensure_data(path: str, total: int = DEFAULT_TOTAL) -> None:
    data_path = Path(path)
    if not data_path.exists() or len(read_rows(path)) != total:
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
