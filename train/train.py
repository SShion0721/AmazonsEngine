import argparse
import os
import struct
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import (
    AmazonsDataset,
    RawCollate,
    FEATURE_SIZE,
    LINE_FEATURE_HASH,
    GLOBAL_FEATURE_SIZE,
    PHASE_BUCKETS,
)

BASE_ACC_SIZE = 512
LINE_ACC_SIZE = 64
HIDDEN_SIZE = 64
SPARSE_INPUT_SIZE = 2 * BASE_ACC_SIZE + 2 * LINE_ACC_SIZE
FT_QUANT_ONE = 127.0
HIDDEN_WEIGHT_SCALE = 64.0
OUTPUT_WEIGHT_SCALE = 16.0
WEIGHT_MAGIC = b"AMZNUE2\0"
LINE_SEED_BASE = 0xA17A5015BEE5C0DE


def parse_csv_arg(text):
    if text is None:
        return None
    values = [item.strip() for item in str(text).split(",")]
    values = [item for item in values if item]
    return values if values else None


def parse_weight_list(text):
    values = parse_csv_arg(text)
    if values is None:
        return None
    weights = [float(v) for v in values]
    if any(w <= 0 for w in weights):
        raise ValueError("--mix-weights must all be positive")
    return weights


def current_model_meta():
    return {
        "arch": "AmazonsNNUE-v2",
        "feature_size": FEATURE_SIZE,
        "base_acc_size": BASE_ACC_SIZE,
        "line_hash": LINE_FEATURE_HASH,
        "line_acc_size": LINE_ACC_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "global_feature_size": GLOBAL_FEATURE_SIZE,
        "phase_buckets": PHASE_BUCKETS,
    }


def ensure_feature_file(raw_path, feature_path, converter_path, skip_convert=False, force_convert=False):
    raw_path = Path(raw_path)
    feature_path = Path(feature_path)
    converter_path = Path(converter_path)

    if skip_convert:
        return feature_path

    needs_convert = force_convert or not feature_path.exists()
    if feature_path.exists() and raw_path.exists():
        needs_convert = needs_convert or os.path.getmtime(feature_path) < os.path.getmtime(raw_path)

    if not needs_convert:
        return feature_path

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw selfplay data not found: {raw_path}")
    if not converter_path.exists():
        raise FileNotFoundError(f"convert_selfplay executable not found: {converter_path}")

    feature_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(converter_path), str(raw_path), str(feature_path)]
    print("Converting selfplay data:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return feature_path


class AmazonsNNUEv2(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_l0 = nn.Linear(FEATURE_SIZE, BASE_ACC_SIZE)
        self.line_l0 = nn.Embedding(LINE_FEATURE_HASH, LINE_ACC_SIZE)
        self.line_l0_bias = nn.Parameter(torch.zeros(LINE_ACC_SIZE))
        self.l1 = nn.Linear(SPARSE_INPUT_SIZE, HIDDEN_SIZE)
        self.global_proj = nn.Linear(GLOBAL_FEATURE_SIZE, HIDDEN_SIZE, bias=False)
        self.phase_bias = nn.Embedding(PHASE_BUCKETS, HIDDEN_SIZE)
        self.l2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.score_head = nn.Linear(HIDDEN_SIZE, 1)
        self.wdl_head = nn.Linear(HIDDEN_SIZE, 1)

    def _line_acc(self, indices, signs):
        emb = self.line_l0(indices)
        return self.line_l0_bias + (emb * signs.unsqueeze(-1)).sum(dim=1)

    def forward(self, base_us, base_them, line_idx_us, line_sign_us,
                line_idx_them, line_sign_them, global_features, phase):
        acc_us = torch.clamp(self.base_l0(base_us), 0.0, 1.0)
        acc_them = torch.clamp(self.base_l0(base_them), 0.0, 1.0)
        line_us = torch.clamp(self._line_acc(line_idx_us, line_sign_us), 0.0, 1.0)
        line_them = torch.clamp(self._line_acc(line_idx_them, line_sign_them), 0.0, 1.0)

        sparse = torch.cat([acc_us, acc_them, line_us, line_them], dim=1)
        h1 = self.l1(sparse) + self.global_proj(global_features) + self.phase_bias(phase)
        h1 = torch.clamp(h1, 0.0, 1.0)
        h2 = torch.clamp(self.l2(h1), 0.0, 1.0)
        return self.score_head(h2), self.wdl_head(h2)


def _write_int_array(f, arr, fmt, scale, lo, hi):
    flat = np.asarray(arr, dtype=np.float64).reshape(-1) * scale
    for value in flat:
        f.write(struct.pack(fmt, int(np.clip(round(value), lo, hi))))


def export_weights(model, path, output_scale):
    scale_hidden_bias = FT_QUANT_ONE * HIDDEN_WEIGHT_SCALE
    scale_output_bias = OUTPUT_WEIGHT_SCALE * output_scale
    scale_output_weight = OUTPUT_WEIGHT_SCALE * output_scale / FT_QUANT_ONE

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack(
            "<8sIIIIIIIIIIIIII",
            WEIGHT_MAGIC,
            2,
            FEATURE_SIZE,
            BASE_ACC_SIZE,
            LINE_FEATURE_HASH,
            LINE_ACC_SIZE,
            HIDDEN_SIZE,
            GLOBAL_FEATURE_SIZE,
            PHASE_BUCKETS,
            SPARSE_INPUT_SIZE,
            int(FT_QUANT_ONE),
            int(HIDDEN_WEIGHT_SCALE),
            int(OUTPUT_WEIGHT_SCALE),
            LINE_SEED_BASE & 0xFFFFFFFF,
            LINE_SEED_BASE >> 32,
        ))

        _write_int_array(f, model.base_l0.bias.detach().cpu().numpy(), "<h", FT_QUANT_ONE, -32768, 32767)
        _write_int_array(f, model.base_l0.weight.detach().cpu().numpy().T, "<h", FT_QUANT_ONE, -32768, 32767)
        _write_int_array(f, model.line_l0_bias.detach().cpu().numpy(), "<h", FT_QUANT_ONE, -32768, 32767)
        _write_int_array(f, model.line_l0.weight.detach().cpu().numpy(), "<h", FT_QUANT_ONE, -32768, 32767)

        _write_int_array(f, model.l1.bias.detach().cpu().numpy(), "<i", scale_hidden_bias, -2147483648, 2147483647)
        _write_int_array(f, model.l1.weight.detach().cpu().numpy(), "<b", HIDDEN_WEIGHT_SCALE, -128, 127)
        _write_int_array(f, model.global_proj.weight.detach().cpu().numpy().T, "<b", HIDDEN_WEIGHT_SCALE, -128, 127)
        _write_int_array(f, model.phase_bias.weight.detach().cpu().numpy(), "<i", scale_hidden_bias, -2147483648, 2147483647)

        _write_int_array(f, model.l2.bias.detach().cpu().numpy(), "<i", scale_hidden_bias, -2147483648, 2147483647)
        _write_int_array(f, model.l2.weight.detach().cpu().numpy(), "<b", HIDDEN_WEIGHT_SCALE, -128, 127)

        _write_int_array(f, model.score_head.bias.detach().cpu().numpy(), "<i", scale_output_bias, -2147483648, 2147483647)
        _write_int_array(f, model.score_head.weight.detach().cpu().numpy().reshape(-1), "<b", scale_output_weight, -128, 127)

    print(f"Exported AMZNUE2 weights: {path}")


def save_checkpoint(model, optimizer, epoch, avg_loss, checkpoint_path, training_meta):
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "avg_loss": float(avg_loss),
        "model_meta": current_model_meta(),
        "training_meta": training_meta,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path} (epoch={epoch + 1}, avg_loss={avg_loss:.6f})")


def load_checkpoint_if_available(model, optimizer, checkpoint_path, device):
    if checkpoint_path is None or not checkpoint_path.exists():
        print(f"No checkpoint found at: {checkpoint_path}")
        return 0
    ckpt = torch.load(checkpoint_path, map_location=device)
    saved_meta = ckpt.get("model_meta")
    if saved_meta is not None and saved_meta != current_model_meta():
        raise RuntimeError(f"Checkpoint architecture mismatch: saved={saved_meta}, current={current_model_meta()}")
    model.load_state_dict(ckpt["model_state_dict"])
    if "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    next_epoch = int(ckpt.get("epoch", -1)) + 1
    print(f"Resumed from checkpoint: {checkpoint_path} (next_epoch={next_epoch + 1})")
    return next_epoch


def load_single_training_dataset(feature_bin_path):
    return AmazonsDataset(str(feature_bin_path))


def build_mixed_dataset_and_sampler(feature_bin_paths, mix_weights=None, samples_per_epoch=0):
    datasets = [load_single_training_dataset(path) for path in feature_bin_paths]
    datasets = [ds for ds in datasets if len(ds) > 0]
    if not datasets:
        return None, None
    if len(datasets) == 1:
        return datasets[0], None

    sizes = [len(ds) for ds in datasets]
    if mix_weights is None:
        mix_weights = [1.0] * len(datasets)
    if len(mix_weights) != len(datasets):
        raise ValueError("--mix-weights count must match data sources")
    total = sum(mix_weights)
    ratios = [w / total for w in mix_weights]
    per_sample = []
    for ratio, size in zip(ratios, sizes):
        per_sample.extend([ratio / size] * size)

    concat = ConcatDataset(datasets)
    if samples_per_epoch <= 0:
        samples_per_epoch = len(concat)
    sampler = WeightedRandomSampler(torch.DoubleTensor(per_sample), samples_per_epoch, replacement=True)
    print(f"Mix sources: {len(datasets)} | sizes={sizes} | ratios={[round(x, 4) for x in ratios]}")
    return concat, sampler


def train(
    raw_bin_path,
    feature_bin_path,
    converter_path,
    nnue_out_path,
    epochs=50,
    checkpoint_path=None,
    save_every=5,
    resume=False,
    export_only=False,
    mix_raw_bin_paths=None,
    mix_feature_bin_paths=None,
    mix_weights=None,
    samples_per_epoch=0,
    skip_convert=False,
    force_convert=False,
    score_scale=500.0,
    score_clip=2000.0,
    output_scale=0.0,
    wdl_loss_weight=0.0,
    residual_loss_weight=0.25,
    augment_d8=True,
    batch_size=8192,
    lr=0.001,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
    if output_scale <= 0.0:
        output_scale = score_scale

    model = AmazonsNNUEv2().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    if export_only:
        if checkpoint_path is None:
            raise RuntimeError("--export-only requires --checkpoint")
        load_checkpoint_if_available(model, optimizer, checkpoint_path, device)
        export_weights(model, nnue_out_path, output_scale)
        return

    if mix_raw_bin_paths:
        feature_paths = []
        for index, raw_path in enumerate(mix_raw_bin_paths):
            feature_path = mix_feature_bin_paths[index]
            feature_paths.append(ensure_feature_file(raw_path, feature_path, converter_path, skip_convert, force_convert))
        dataset, sampler = build_mixed_dataset_and_sampler(feature_paths, mix_weights, samples_per_epoch)
    else:
        feature_path = ensure_feature_file(raw_bin_path, feature_bin_path, converter_path, skip_convert, force_convert)
        dataset = load_single_training_dataset(feature_path)
        sampler = None

    if dataset is None or len(dataset) == 0:
        print("No training samples available.")
        return

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=RawCollate(augment_d8=augment_d8),
        pin_memory=(device.type == "cuda"),
        num_workers=0 if os.name == "nt" else 8,
    )

    score_criterion = nn.SmoothL1Loss(beta=0.25)
    wdl_criterion = nn.BCEWithLogitsLoss()
    print(f"Model: base 300->{BASE_ACC_SIZE}, line {LINE_FEATURE_HASH}->{LINE_ACC_SIZE}, sparse {SPARSE_INPUT_SIZE}->{HIDDEN_SIZE}")
    print(f"Loss: score + {residual_loss_weight}*residual + {wdl_loss_weight}*wdl | score_scale={score_scale}")

    training_meta = {
        "score_scale": score_scale,
        "score_clip": score_clip,
        "output_scale": output_scale,
        "wdl_loss_weight": wdl_loss_weight,
        "residual_loss_weight": residual_loss_weight,
        "augment_d8": augment_d8,
        "batch_size": batch_size,
        "lr": lr,
    }

    start_epoch = load_checkpoint_if_available(model, optimizer, checkpoint_path, device) if resume else 0
    best_loss = float("inf")
    log_root = checkpoint_path.parent if checkpoint_path is not None else nnue_out_path.parent
    writer = SummaryWriter(log_dir=str(log_root / "runs" / "nnue_v2"))

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch in pbar:
            (
                base_us, base_them,
                line_idx_us, line_sign_us,
                line_idx_them, line_sign_them,
                global_features, phase,
                classical, target, score,
            ) = batch

            base_us = base_us.to(device, non_blocking=True).float()
            base_them = base_them.to(device, non_blocking=True).float()
            line_idx_us = line_idx_us.to(device, non_blocking=True).long()
            line_sign_us = line_sign_us.to(device, non_blocking=True).float()
            line_idx_them = line_idx_them.to(device, non_blocking=True).long()
            line_sign_them = line_sign_them.to(device, non_blocking=True).float()
            global_features = global_features.to(device, non_blocking=True).float()
            phase = phase.to(device, non_blocking=True).long()
            classical = classical.to(device, non_blocking=True).float()
            target = target.to(device, non_blocking=True).float()
            score = score.to(device, non_blocking=True).float()

            if score_clip > 0.0:
                score = torch.clamp(score, -score_clip, score_clip)

            score_target = score / score_scale
            residual_target = (score - classical) / score_scale

            optimizer.zero_grad(set_to_none=True)
            pred, wdl = model(
                base_us, base_them,
                line_idx_us, line_sign_us,
                line_idx_them, line_sign_them,
                global_features, phase,
            )
            loss = score_criterion(pred, score_target)
            if residual_loss_weight > 0.0:
                loss = loss + residual_loss_weight * score_criterion(pred - classical / score_scale, residual_target)
            if wdl_loss_weight > 0.0:
                loss = loss + wdl_loss_weight * wdl_criterion(wdl, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(loader)
        writer.add_scalar("Loss/train", avg_loss, epoch)

        if checkpoint_path is not None and save_every > 0:
            if (epoch + 1) % save_every == 0 or (epoch + 1) == epochs:
                save_checkpoint(model, optimizer, epoch, avg_loss, checkpoint_path, training_meta)

        if avg_loss < best_loss:
            best_loss = avg_loss
            if checkpoint_path is not None:
                save_checkpoint(model, optimizer, epoch, avg_loss, checkpoint_path.parent / "best.ckpt", training_meta)
            export_weights(model, nnue_out_path.parent / f"{nnue_out_path.stem}_best.bin", output_scale)
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.6f} (new best)")

    writer.close()
    export_weights(model, nnue_out_path, output_scale)
    print(f"Training complete. Best loss: {best_loss:.6f}")


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent
    default_raw_bin = repo_root / "selfplay_data.bin"
    default_feature_bin = repo_root / "selfplay_features_v2.bin"
    default_converter = repo_root / "build" / "convert_selfplay.exe"
    default_nnue_out = repo_root / "nnue_v2.bin"

    parser = argparse.ArgumentParser(description="Train AmazonsNNUE-v2 from selfplay data.")
    parser.add_argument("--raw-bin", type=Path, default=default_raw_bin)
    parser.add_argument("--feature-bin", type=Path, default=default_feature_bin)
    parser.add_argument("--mix-raw-bins", type=str, default=None)
    parser.add_argument("--mix-feature-bins", type=str, default=None)
    parser.add_argument("--mix-weights", type=str, default=None)
    parser.add_argument("--samples-per-epoch", type=int, default=0)
    parser.add_argument("--converter", type=Path, default=default_converter)
    parser.add_argument("--nnue-out", type=Path, default=default_nnue_out)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--checkpoint", type=Path, default=repo_root / "train_v2_last.ckpt")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--skip-convert", action="store_true")
    parser.add_argument("--force-convert", action="store_true")
    parser.add_argument("--score-scale", type=float, default=500.0)
    parser.add_argument("--score-clip", type=float, default=2000.0)
    parser.add_argument("--output-scale", type=float, default=0.0)
    parser.add_argument("--wdl-loss-weight", type=float, default=0.0)
    parser.add_argument("--residual-loss-weight", type=float, default=0.25)
    parser.add_argument("--augment-d8", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    mix_raw_values = parse_csv_arg(args.mix_raw_bins)
    mix_feature_values = parse_csv_arg(args.mix_feature_bins)
    mix_weights = parse_weight_list(args.mix_weights)

    mix_raw_bin_paths = [Path(v).resolve() for v in mix_raw_values] if mix_raw_values else None
    mix_feature_bin_paths = None
    if mix_raw_bin_paths:
        if mix_feature_values:
            if len(mix_feature_values) != len(mix_raw_bin_paths):
                raise ValueError("--mix-feature-bins count must match --mix-raw-bins")
            mix_feature_bin_paths = [Path(v).resolve() for v in mix_feature_values]
        else:
            mix_feature_bin_paths = [path.with_suffix(".features_v2.bin") for path in mix_raw_bin_paths]

    train(
        raw_bin_path=args.raw_bin.resolve(),
        feature_bin_path=args.feature_bin.resolve(),
        converter_path=args.converter.resolve(),
        nnue_out_path=args.nnue_out.resolve(),
        epochs=args.epochs,
        checkpoint_path=args.checkpoint.resolve() if args.checkpoint else None,
        save_every=args.save_every,
        resume=args.resume,
        export_only=args.export_only,
        mix_raw_bin_paths=mix_raw_bin_paths,
        mix_feature_bin_paths=mix_feature_bin_paths,
        mix_weights=mix_weights,
        samples_per_epoch=args.samples_per_epoch,
        skip_convert=args.skip_convert,
        force_convert=args.force_convert,
        score_scale=args.score_scale,
        score_clip=args.score_clip,
        output_scale=args.output_scale,
        wdl_loss_weight=args.wdl_loss_weight,
        residual_loss_weight=args.residual_loss_weight,
        augment_d8=args.augment_d8,
        batch_size=args.batch_size,
        lr=args.lr,
    )
