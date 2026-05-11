import argparse
import os
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

BOARD_SQ = 100
FEATURE_SIZE = 300
AMZSV3_MAGIC = b"AMZSV3\0\0"
AMZPOL1_MAGIC = b"AMZPOL1\0"


def move_from(move):
    return (int(move) >> 14) & 0x7F


def move_to(move):
    return (int(move) >> 7) & 0x7F


def move_arrow(move):
    return int(move) & 0x7F


def side_relative_features(board, side):
    feat = np.zeros(FEATURE_SIZE, dtype=np.float32)
    for sq, ch in enumerate(board):
        if ch == ord("0"):
            continue
        if ch == ord("3"):
            feat[200 + sq] = 1.0
        elif (ch == ord("1") and side == 0) or (ch == ord("2") and side == 1):
            feat[sq] = 1.0
        else:
            feat[100 + sq] = 1.0
    return feat


class PolicyVisitDataset(Dataset):
    def __init__(self, path, max_records=0):
        self.records = []
        path = Path(path)
        with path.open("rb") as f:
            header = f.read(20)
            if len(header) != 20:
                raise RuntimeError(f"{path} is too small for AMZSV3")
            magic, version, board_squares, _max_topk = struct.unpack("<8sIII", header)
            if magic != AMZSV3_MAGIC or version != 1 or board_squares != BOARD_SQ:
                raise RuntimeError(f"{path} is not a compatible AMZSV3 file")

            while True:
                fixed = f.read(110)
                if not fixed:
                    break
                if len(fixed) != 110:
                    raise RuntimeError("Truncated AMZSV3 fixed record")
                side, board, outcome, score, best_move, visit_count = struct.unpack("<b100sbhIH", fixed)
                visits_raw = f.read(visit_count * 6)
                if len(visits_raw) != visit_count * 6:
                    raise RuntimeError("Truncated AMZSV3 visit list")
                visits = []
                for i in range(visit_count):
                    move, count = struct.unpack_from("<IH", visits_raw, i * 6)
                    if count > 0 and move != 0:
                        visits.append((move, count))
                if visits:
                    self.records.append((side, board, outcome, score, best_move, visits))
                if max_records > 0 and len(self.records) >= max_records:
                    break

        print(f"Loaded {len(self.records)} AMZSV3 policy records from {path}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        side, board, _outcome, _score, _best_move, visits = self.records[idx]
        feat = side_relative_features(board, side)
        target_from = np.zeros(BOARD_SQ, dtype=np.float32)
        target_to = np.zeros(BOARD_SQ, dtype=np.float32)
        target_arrow = np.zeros(BOARD_SQ, dtype=np.float32)
        total = float(sum(count for _move, count in visits))
        for move, count in visits:
            w = float(count) / total
            f = move_from(move)
            t = move_to(move)
            a = move_arrow(move)
            if 0 <= f < BOARD_SQ and 0 <= t < BOARD_SQ and 0 <= a < BOARD_SQ:
                target_from[f] += w
                target_to[t] += w
                target_arrow[a] += w
        return feat, target_from, target_to, target_arrow


class FactorPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.from_head = nn.Linear(FEATURE_SIZE, BOARD_SQ)
        self.to_head = nn.Linear(FEATURE_SIZE, BOARD_SQ)
        self.arrow_head = nn.Linear(FEATURE_SIZE, BOARD_SQ)

    def forward(self, x):
        return self.from_head(x), self.to_head(x), self.arrow_head(x)


def soft_cross_entropy(logits, target):
    logp = torch.log_softmax(logits, dim=1)
    return -(target * logp).sum(dim=1).mean()


def export_amzpol1(model, path, scale):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    heads = [model.from_head, model.to_head, model.arrow_head]

    bias = np.zeros((3, BOARD_SQ), dtype=np.int16)
    weights = np.zeros((3, FEATURE_SIZE, BOARD_SQ), dtype=np.int8)
    for h, head in enumerate(heads):
        b = head.bias.detach().cpu().numpy() * scale
        w = head.weight.detach().cpu().numpy().T * scale
        bias[h] = np.clip(np.rint(b), -32768, 32767).astype(np.int16)
        weights[h] = np.clip(np.rint(w), -128, 127).astype(np.int8)

    with path.open("wb") as f:
        f.write(struct.pack("<8sIII", AMZPOL1_MAGIC, 1, BOARD_SQ, FEATURE_SIZE))
        f.write(bias.tobytes(order="C"))
        f.write(weights.tobytes(order="C"))
    print(f"Exported AMZPOL1 policy prior: {path}")


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = PolicyVisitDataset(args.policy_visits, args.max_records)
    if len(dataset) == 0:
        raise RuntimeError("No AMZSV3 policy records found")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0 if os.name == "nt" else args.workers,
        pin_memory=(device.type == "cuda"),
    )

    model = FactorPolicy().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        pbar = tqdm(loader, desc=f"Policy epoch {epoch + 1}/{args.epochs}")
        for feat, tf, tt, ta in pbar:
            feat = feat.to(device, non_blocking=True).float()
            tf = tf.to(device, non_blocking=True).float()
            tt = tt.to(device, non_blocking=True).float()
            ta = ta.to(device, non_blocking=True).float()

            lf, lt, la = model(feat)
            loss = (soft_cross_entropy(lf, tf)
                    + soft_cross_entropy(lt, tt)
                    + soft_cross_entropy(la, ta)) / 3.0
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total += float(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        print(f"Epoch {epoch + 1}: loss={total / max(1, len(loader)):.6f}")

    export_amzpol1(model, args.out, args.quant_scale)


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Train/export AMZPOL1 CPU policy prior from AMZSV3 visits.")
    parser.add_argument("--policy-visits", type=Path, default=repo_root / "selfplay_policy_visits.amzsv3")
    parser.add_argument("--out", type=Path, default=repo_root / "policy.bin")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--quant-scale", type=float, default=64.0)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    train(parser.parse_args())
