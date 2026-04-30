import random
from pathlib import Path


PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
TOKENS = [PAD, BOS, EOS, "+", "="] + list("0123456789")
TOKEN_TO_ID = {token: idx for idx, token in enumerate(TOKENS)}
ID_TO_TOKEN = {idx: token for token, idx in TOKEN_TO_ID.items()}


def make_addition(a: int, b: int) -> str:
    return f"{a}+{b}={a + b}"


def generate_data(path: str, total: int = 10000, seed: int = 42) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    rows = []
    seen = set()
    while len(rows) < total:
        a = random.randint(0, 9999)
        b = random.randint(0, 9999 - a)
        row = make_addition(a, b)
        if row not in seen:
            seen.add(row)
            rows.append(row)
    target.write_text("\n".join(rows) + "\n", encoding="utf-8")


def ensure_data(path: str, total: int = 10000) -> None:
    if not Path(path).exists():
        generate_data(path, total=total)


def read_rows(path: str) -> list[str]:
    return Path(path).read_text(encoding="utf-8").splitlines()


def split_row(row: str) -> tuple[str, str]:
    expr, answer = row.split("=")
    return expr + "=", answer


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


class AdditionDataset:
    def __init__(self, rows: list[str]):
        self.samples = [split_row(row) for row in rows]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        import torch

        expr, answer = self.samples[idx]
        src = encode_text(expr)
        tgt_in = [TOKEN_TO_ID[BOS]] + encode_text(answer)
        tgt_out = encode_text(answer) + [TOKEN_TO_ID[EOS]]
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
