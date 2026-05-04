import argparse
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import AmazonsDataset, RawCollate

FEATURE_SIZE = 300
ACC_SIZE = 512
HIDDEN_SIZE = 64
FT_QUANT_ONE = 127.0
HIDDEN_WEIGHT_SCALE = 64.0
OUTPUT_WEIGHT_SCALE = 16.0


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

    try:
        weights = [float(v) for v in values]
    except ValueError:
        print(f"Invalid --mix-weights: {text}")
        return None

    if any(weight <= 0 for weight in weights):
        print(f"All mix weights must be positive: {weights}")
        return None
    return weights


def default_feature_path_for_raw(raw_path):
    return raw_path.with_suffix(".features.bin")


def current_model_meta():
    return {
        "feature_size": FEATURE_SIZE,
        "acc_size": ACC_SIZE,
        "hidden_size": HIDDEN_SIZE,
    }


def describe_target_mode(train_target, score_loss=None, blend_lambda=0.0, score_scale=400.0, score_clip=0.0):
    if train_target == "wdl":
        print(f"Training Target: blended-wdl | lambda={blend_lambda} | score_scale={score_scale}")
        print(f"  Target = {blend_lambda}*Outcome + {1.0 - blend_lambda}*Sigmoid(Score/{score_scale})")
        return

    clip_desc = "disabled" if score_clip <= 0.0 else f"+/-{score_clip}"
    print(f"Training Target: score-regression | loss={score_loss} | score_scale={score_scale} | score_clip={clip_desc}")
    print(f"  Target = clamp(Score, -clip, +clip) / {score_scale}")


class AmazonsNNUE(nn.Module):
    def __init__(self):
        super().__init__()
        self.l0 = nn.Linear(FEATURE_SIZE, ACC_SIZE)
        self.l1 = nn.Linear(ACC_SIZE * 2, HIDDEN_SIZE)
        self.l2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.l3 = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, us, them):
        acc_us = self.l0(us)
        acc_them = self.l0(them)

        act_us = torch.clamp(acc_us, 0.0, 1.0)
        act_them = torch.clamp(acc_them, 0.0, 1.0)

        concat = torch.cat([act_us, act_them], dim=1)
        act1 = torch.clamp(self.l1(concat), 0.0, 1.0)
        act2 = torch.clamp(self.l2(act1), 0.0, 1.0)
        return self.l3(act2)


def export_weights(model, path, output_scale):
    # Quantize and export into the compact C++ layout.
    #
    # With score-regression training, model output approximates score/output_scale.
    # With WDL training, model output is a logit that we reinterpret into engine
    # score units at export time. In both cases output_scale defines how the raw
    # float output maps back to the engine score domain.
    scale_hidden_bias = FT_QUANT_ONE * HIDDEN_WEIGHT_SCALE
    scale_output_bias = OUTPUT_WEIGHT_SCALE * output_scale
    scale_output_weight = OUTPUT_WEIGHT_SCALE * output_scale / FT_QUANT_ONE

    print("Packing weights into memory layout aligned with C++ 'NetworkParameters'...")
    print(f"  output_scale={output_scale} (engine score units per network output)")
    with open(path, "wb") as f:
        l0_b = model.l0.bias.detach().cpu().numpy().flatten() * FT_QUANT_ONE
        for b in l0_b:
            f.write(struct.pack("<h", int(np.clip(b, -32768, 32767))))

        l0_w = model.l0.weight.detach().cpu().numpy().T * FT_QUANT_ONE
        for row in l0_w:
            for w in row:
                f.write(struct.pack("<h", int(np.clip(w, -32768, 32767))))

        l1_b = model.l1.bias.detach().cpu().numpy().flatten() * scale_hidden_bias
        for b in l1_b:
            f.write(struct.pack("<i", int(np.clip(b, -2147483648, 2147483647))))

        l1_w = model.l1.weight.detach().cpu().numpy().T * HIDDEN_WEIGHT_SCALE
        for row in l1_w:
            for w in row:
                f.write(struct.pack("<b", int(np.clip(w, -128, 127))))

        l2_b = model.l2.bias.detach().cpu().numpy().flatten() * scale_hidden_bias
        for b in l2_b:
            f.write(struct.pack("<i", int(np.clip(b, -2147483648, 2147483647))))

        l2_w = model.l2.weight.detach().cpu().numpy().T * HIDDEN_WEIGHT_SCALE
        for row in l2_w:
            for w in row:
                f.write(struct.pack("<b", int(np.clip(w, -128, 127))))

        l3_b = model.l3.bias.detach().cpu().numpy().flatten() * scale_output_bias
        for b in l3_b:
            f.write(struct.pack("<i", int(np.clip(b, -2147483648, 2147483647))))

        l3_w = model.l3.weight.detach().cpu().numpy().T * scale_output_weight
        for row in l3_w:
            for w in row:
                f.write(struct.pack("<b", int(np.clip(w, -128, 127))))


def save_checkpoint(model, optimizer, epoch, avg_loss, checkpoint_path, training_meta):
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "avg_loss": float(avg_loss),
        "model_meta": current_model_meta(),
        "training_meta": training_meta,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(payload, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path} (epoch={epoch + 1}, avg_loss={avg_loss:.6f})")


def load_checkpoint_if_available(model, optimizer, checkpoint_path, device):
    if not checkpoint_path.exists():
        print(f"No checkpoint found at: {checkpoint_path}")
        return 0

    ckpt = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" not in ckpt:
        print(f"Checkpoint missing model_state_dict: {checkpoint_path}")
        return 0

    saved_meta = ckpt.get("model_meta")
    current_meta = current_model_meta()
    if saved_meta is not None and saved_meta != current_meta:
        raise RuntimeError(
            f"Checkpoint architecture mismatch: saved={saved_meta}, current={current_meta}. "
            "This usually means the checkpoint was trained with the old 256/32/32 network and "
            "cannot be resumed/exported with the new 512/64/64 build."
        )

    try:
        model.load_state_dict(ckpt["model_state_dict"])
    except RuntimeError as exc:
        raise RuntimeError(
            f"Failed to load checkpoint '{checkpoint_path}'. The saved tensor shapes do not match "
            f"the current architecture {current_meta}. Old checkpoints from the smaller network "
            "need a fresh retrain under the new architecture."
        ) from exc

    if "optimizer_state_dict" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except ValueError:
            print(f"Warning: optimizer state in {checkpoint_path} was skipped due to shape mismatch.")

    last_epoch = int(ckpt.get("epoch", -1))
    next_epoch = last_epoch + 1
    print(f"Resumed from checkpoint: {checkpoint_path} (last_epoch={last_epoch + 1})")
    return next_epoch


def load_single_training_dataset(raw_bin_path):
    return AmazonsDataset(str(raw_bin_path))


def build_mixed_dataset_and_sampler(
    raw_bin_paths,
    feature_bin_paths,
    converter_path,
    convert_mode,
    skip_convert=False,
    force_convert=False,
    mix_weights=None,
    samples_per_epoch=0,
):
    datasets = []
    dataset_sizes = []
    for index, raw_bin_path in enumerate(raw_bin_paths):
        feature_bin_path = feature_bin_paths[index]
        dataset = load_single_training_dataset(raw_bin_path=raw_bin_path)
        if dataset is None:
            return None, None
        if len(dataset) == 0:
            print(f"Warning: empty dataset skipped: {raw_bin_path}")
            continue
        datasets.append(dataset)
        dataset_sizes.append(len(dataset))

    if not datasets:
        print("No usable datasets found for mix training.")
        return None, None

    if len(datasets) == 1:
        print(f"Mix requested but only one non-empty source, using single dataset ({dataset_sizes[0]} samples).")
        return datasets[0], None

    if mix_weights is None:
        mix_weights = [1.0] * len(datasets)

    if len(mix_weights) != len(datasets):
        print(f"Mix weights count mismatch: weights={len(mix_weights)} sources={len(datasets)}")
        return None, None

    weight_sum = sum(mix_weights)
    normalized = [weight / weight_sum for weight in mix_weights]
    per_sample_weights = []
    for source_weight, source_size in zip(normalized, dataset_sizes):
        per_sample_weights.extend([source_weight / source_size] * source_size)

    concat_dataset = ConcatDataset(datasets)
    if samples_per_epoch <= 0:
        samples_per_epoch = len(concat_dataset)

    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(per_sample_weights),
        num_samples=samples_per_epoch,
        replacement=True,
    )

    print(f"Mix sources: {len(datasets)} | sizes={dataset_sizes}")
    print(f"Mix ratios : {[round(x, 4) for x in normalized]} | samples/epoch={samples_per_epoch}")
    return concat_dataset, sampler


def make_criterion(train_target, score_loss):
    if train_target == "wdl":
        return nn.BCEWithLogitsLoss()
    if score_loss == "mse":
        return nn.MSELoss()
    if score_loss == "smoothl1":
        return nn.SmoothL1Loss(beta=0.25)
    raise ValueError(f"Unsupported score loss: {score_loss}")


def train(
    raw_bin_path,
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
    train_target="score",
    score_loss="smoothl1",
    blend_lambda=0.5,
    score_scale=400.0,
    score_clip=2000.0,
    output_scale=0.0,
    augment_d8=True,
    batch_size=32768,
    lr=0.005,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True

    if output_scale <= 0.0:
        output_scale = score_scale
    print(f"Export Output Scale: {output_scale}")

    model = AmazonsNNUE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if export_only:
        if checkpoint_path is None:
            print("Export-only requested but no checkpoint path was provided.")
            return
        load_checkpoint_if_available(model, optimizer, checkpoint_path, device)
        nnue_out_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Exporting checkpoint weights to: {nnue_out_path}")
        export_weights(model, nnue_out_path, output_scale)
        print(f"Done! You can now load '{nnue_out_path}' in the C++ engine.")
        return

    sampler = None
    if mix_raw_bin_paths:
        dataset, sampler = build_mixed_dataset_and_sampler(
            raw_bin_paths=mix_raw_bin_paths,
            feature_bin_paths=mix_feature_bin_paths,
            converter_path=None,
            convert_mode="file",
            skip_convert=True,
            force_convert=False,
            mix_weights=mix_weights,
            samples_per_epoch=samples_per_epoch,
        )
        if dataset is None:
            return
    else:
        dataset = load_single_training_dataset(raw_bin_path=raw_bin_path)
        if dataset is None:
            return

    if len(dataset) == 0:
        return

    collate_fn = RawCollate(augment_d8=augment_d8)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
        num_workers=14,
        prefetch_factor=2,
        persistent_workers=True if device.type == "cuda" else False,
    )

    criterion = make_criterion(train_target, score_loss)
    print(f"Model: {FEATURE_SIZE} -> {ACC_SIZE}x2 -> {HIDDEN_SIZE} -> {HIDDEN_SIZE} -> 1")
    print(f"D8 augmentation: {'on' if augment_d8 else 'off'}")
    describe_target_mode(
        train_target,
        score_loss=score_loss,
        blend_lambda=blend_lambda,
        score_scale=score_scale,
        score_clip=score_clip,
    )

    training_meta = {
        "train_target": train_target,
        "score_loss": score_loss,
        "blend_lambda": blend_lambda,
        "score_scale": score_scale,
        "score_clip": score_clip,
        "output_scale": output_scale,
        "augment_d8": augment_d8,
        "batch_size": batch_size,
        "lr": lr,
    }

    start_epoch = 0
    best_loss = float("inf")

    log_root = checkpoint_path.parent if checkpoint_path is not None else nnue_out_path.parent
    log_dir = log_root / "runs" / "nnue_experiment"
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"TensorBoard logs will be saved to: {log_dir}")
    print("Run 'tensorboard --logdir runs' to view training progress.")

    if resume and checkpoint_path is not None:
        start_epoch = load_checkpoint_if_available(model, optimizer, checkpoint_path, device)
        if start_epoch >= epochs:
            print(f"Checkpoint already at epoch {start_epoch}, target epochs={epochs}.")
            print(f"Skipping training loop and exporting current weights to: {nnue_out_path}")
            nnue_out_path.parent.mkdir(parents=True, exist_ok=True)
            export_weights(model, nnue_out_path, output_scale)
            print(f"Done! You can now resume the C++ engine to load '{nnue_out_path}'.")
            return

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for b_us, b_them, b_target, b_score in pbar:
            optimizer.zero_grad()

            b_us = b_us.to(device, non_blocking=True).float()
            b_them = b_them.to(device, non_blocking=True).float()
            pred = model(b_us, b_them)

            if train_target == "wdl":
                b_target = b_target.to(device, non_blocking=True).float()
                b_score = b_score.to(device, non_blocking=True).float()
                score_win_prob = torch.sigmoid(b_score / score_scale)
                target = blend_lambda * b_target + (1.0 - blend_lambda) * score_win_prob
            else:
                b_score = b_score.to(device, non_blocking=True).float()
                if score_clip > 0.0:
                    b_score = torch.clamp(b_score, -score_clip, score_clip)
                target = b_score / score_scale

            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(loader)
        writer.add_scalar("Loss/train", avg_loss, epoch)

        if checkpoint_path is not None and save_every > 0:
            is_periodic = ((epoch + 1) % save_every == 0)
            is_last = (epoch + 1) == epochs
            if is_periodic or is_last:
                save_checkpoint(model, optimizer, epoch, avg_loss, checkpoint_path, training_meta)

        if avg_loss < best_loss:
            best_loss = avg_loss
            if checkpoint_path is not None:
                best_ckpt_path = checkpoint_path.parent / "best.ckpt"
                save_checkpoint(model, optimizer, epoch, avg_loss, best_ckpt_path, training_meta)
            best_nnue_path = nnue_out_path.parent / f"{nnue_out_path.stem}_best.bin"
            export_weights(model, best_nnue_path, output_scale)
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.6f} (New Best!)")
        elif (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.6f}")

    writer.close()
    print(f"Training complete. Best Loss: {best_loss:.6f}")
    print(f"Exporting final network to: {nnue_out_path}")
    nnue_out_path.parent.mkdir(parents=True, exist_ok=True)
    export_weights(model, nnue_out_path, output_scale)
    print(f"Done! You can now resume the C++ engine to load '{nnue_out_path}'.")


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent
    default_raw_bin = repo_root / "selfplay_data.bin"
    default_feature_bin = repo_root / "selfplay_features.bin"
    default_converter = repo_root / "build" / "convert_selfplay.exe"
    default_nnue_out = repo_root / "nnue.bin"

    parser = argparse.ArgumentParser(description="Train NNUE from selfplay data.")
    parser.add_argument("--raw-bin", type=Path, default=default_raw_bin, help="Path to selfplay_data.bin")
    parser.add_argument("--feature-bin", type=Path, default=default_feature_bin, help="Path to selfplay_features.bin")
    parser.add_argument("--mix-raw-bins", type=str, default=None, help="Comma-separated raw bin list for mixed training")
    parser.add_argument("--mix-feature-bins", type=str, default=None, help="Comma-separated feature bin list (file mode only)")
    parser.add_argument("--mix-weights", type=str, default=None, help="Comma-separated mix weights, e.g. 0.3,0.7")
    parser.add_argument("--samples-per-epoch", type=int, default=0, help="When mixing, sampled examples per epoch (0 means total mixed size)")
    parser.add_argument("--converter", type=Path, default=default_converter, help="Path to C++ converter executable")
    parser.add_argument("--nnue-out", type=Path, default=default_nnue_out, help="Output path for exported nnue.bin")
    parser.add_argument("--convert-mode", choices=["pipe", "file"], default="file",
                        help="pipe: convert in-memory (no feature file); file: write/read selfplay_features.bin")
    parser.add_argument("--epochs", type=int, default=50, help="Total training epochs")
    parser.add_argument("--checkpoint", type=Path, default=repo_root / "train_last.ckpt", help="Checkpoint path")
    parser.add_argument("--save-every", type=int, default=5, help="Save checkpoint every N epochs (<=0 disables periodic saves)")
    parser.add_argument("--resume", action="store_true", help="Resume training from --checkpoint")
    parser.add_argument("--export-only", action="store_true",
                        help="Load --checkpoint and export nnue.bin without running training")
    parser.add_argument("--skip-convert", action="store_true", help="Skip calling C++ converter before training")
    parser.add_argument("--force-convert", action="store_true", help="Force conversion even if feature file looks up-to-date")
    parser.add_argument("--train-target", choices=["score", "wdl"], default="score",
                        help="score: regress engine score directly; wdl: blended win-probability distillation")
    parser.add_argument("--score-loss", choices=["smoothl1", "mse"], default="smoothl1",
                        help="Regression loss when --train-target=score")
    parser.add_argument("--blend-lambda", type=float, default=0.1,
                        help="Blended loss lambda: 0=pure distillation, 1=pure outcome, used only in --train-target=wdl")
    parser.add_argument("--score-scale", type=float, default=500.0,
                        help="Normalization divisor for score regression, or sigmoid divisor for WDL mode")
    parser.add_argument("--score-clip", type=float, default=2000.0,
                        help="Clamp teacher scores before regression (0 disables clipping)")
    parser.add_argument("--output-scale", type=float, default=0.0,
                        help="Engine score units per network output at export time (default: score-scale)")
    parser.add_argument("--augment-d8", action=argparse.BooleanOptionalAction, default=True,
                        help="Apply one random D8 symmetry per batch (default: on)")
    parser.add_argument("--batch-size", type=int, default=16384, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    args = parser.parse_args()

    mix_raw_values = parse_csv_arg(args.mix_raw_bins)
    mix_feature_values = parse_csv_arg(args.mix_feature_bins)
    mix_weights = parse_weight_list(args.mix_weights)

    mix_raw_bin_paths = None
    mix_feature_bin_paths = None
    if mix_raw_values:
        mix_raw_bin_paths = [Path(value).resolve() for value in mix_raw_values]

        if mix_feature_values:
            if len(mix_feature_values) != len(mix_raw_bin_paths):
                raise ValueError("--mix-feature-bins count must match --mix-raw-bins count")
            mix_feature_bin_paths = [Path(value).resolve() for value in mix_feature_values]
        else:
            mix_feature_bin_paths = [default_feature_path_for_raw(path) for path in mix_raw_bin_paths]

        if mix_weights is not None and len(mix_weights) != len(mix_raw_bin_paths):
            raise ValueError("--mix-weights count must match --mix-raw-bins count")

    try:
        train(
            raw_bin_path=args.raw_bin,
            nnue_out_path=args.nnue_out,
            epochs=args.epochs,
            checkpoint_path=args.checkpoint,
            save_every=args.save_every,
            resume=args.resume,
            export_only=args.export_only,
            mix_raw_bin_paths=mix_raw_bin_paths,
            mix_feature_bin_paths=mix_feature_bin_paths,
            mix_weights=mix_weights,
            samples_per_epoch=args.samples_per_epoch,
            train_target=args.train_target,
            score_loss=args.score_loss,
            blend_lambda=args.blend_lambda,
            score_scale=args.score_scale,
            score_clip=args.score_clip,
            output_scale=args.output_scale,
            augment_d8=args.augment_d8,
            batch_size=args.batch_size,
            lr=args.lr,
        )
    except RuntimeError as exc:
        print(f"Error: {exc}")
