import os
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset

FEATURE_SIZE = 300
BOARD_SIZE = 10
BOARD_SQ = 100
NUM_LINES = 58
LINE_FEATURE_HASH = 1 << 16
GLOBAL_FEATURE_SIZE = 32
PHASE_BUCKETS = 8
FEATURE_MAGIC = b"AMZFV2\0\0"


def _splitmix64_stream(count):
    x = np.uint64(0xA17A5015BEE5C0DE)
    out = []
    with np.errstate(over="ignore"):
        for _ in range(count):
            x = np.uint64(x + np.uint64(0x9E3779B97F4A7C15))
            z = x
            z = np.uint64((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9))
            z = np.uint64((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB))
            z = z ^ (z >> np.uint64(31))
            out.append(np.uint32(z & np.uint64(0xFFFFFFFF)))
    return np.asarray(out, dtype=np.uint32)


LINE_SEED = _splitmix64_stream(NUM_LINES)


def _build_sq_lines():
    refs = np.zeros((BOARD_SQ, 4, 2), dtype=np.uint8)
    for sq in range(BOARD_SQ):
        r, c = divmod(sq, BOARD_SIZE)
        refs[sq, 0] = (r, 2 * c)
        refs[sq, 1] = (10 + c, 2 * r)

        diag = r - c + 9
        dr, dc = r, c
        while dr > 0 and dc > 0:
            dr -= 1
            dc -= 1
        refs[sq, 2] = (20 + diag, 2 * (r - dr))

        anti = r + c
        dr, dc = r, c
        while dr > 0 and dc < 9:
            dr -= 1
            dc += 1
        refs[sq, 3] = (39 + anti, 2 * (r - dr))
    return refs


SQ_LINES = _build_sq_lines()


def mix_line_hash(x):
    x = np.asarray(x, dtype=np.uint32)
    with np.errstate(over="ignore"):
        x ^= x >> np.uint32(16)
        x *= np.uint32(0x7FEB352D)
        x ^= x >> np.uint32(15)
        x *= np.uint32(0x846CA68B)
        x ^= x >> np.uint32(16)
    return x.astype(np.uint32)


class AmazonsDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.header_dtype = np.dtype([
            ("magic", "S8"),
            ("version", np.uint32),
            ("board_size", np.uint32),
            ("global_feature_size", np.uint32),
        ])
        self.record_dtype = np.dtype([
            ("side", np.int8),
            ("board", "S100"),
            ("outcome", np.int8),
            ("score", np.int16),
            ("classical", np.int16),
            ("phase_bucket", np.uint8),
            ("global", (np.int8, GLOBAL_FEATURE_SIZE)),
        ])

        self.data_path = str(data_path)
        if not os.path.exists(self.data_path):
            print(f"Feature dataset path {self.data_path} not found.")
            self.num_samples = 0
            return

        stats = os.stat(self.data_path)
        if stats.st_size < self.header_dtype.itemsize:
            print(f"Feature dataset too small: {self.data_path}")
            self.num_samples = 0
            return

        header = np.memmap(self.data_path, dtype=self.header_dtype, mode="r", shape=(1,))[0]
        if bytes(header["magic"]).rstrip(b"\0") != b"AMZFV2" or int(header["version"]) != 2:
            raise RuntimeError(f"{self.data_path} is not an AMZFV2 feature file")
        if int(header["board_size"]) != BOARD_SQ or int(header["global_feature_size"]) != GLOBAL_FEATURE_SIZE:
            raise RuntimeError(f"{self.data_path} AMZFV2 constants do not match the trainer")

        payload = stats.st_size - self.header_dtype.itemsize
        self.num_samples = payload // self.record_dtype.itemsize
        self.data = None
        print(f"Mapped AMZFV2 dataset: {self.data_path} ({self.num_samples} records, ~{stats.st_size / 1024**3:.2f} GB)")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data_path, idx


class RawCollate:
    def __init__(self, augment_d8=False):
        self.header_dtype = np.dtype([
            ("magic", "S8"),
            ("version", np.uint32),
            ("board_size", np.uint32),
            ("global_feature_size", np.uint32),
        ])
        self.record_dtype = np.dtype([
            ("side", np.int8),
            ("board", "S100"),
            ("outcome", np.int8),
            ("score", np.int16),
            ("classical", np.int16),
            ("phase_bucket", np.uint8),
            ("global", (np.int8, GLOBAL_FEATURE_SIZE)),
        ])
        self.handles = {}
        self.augment_d8 = augment_d8

    def _apply_batch_d8(self, board_2d):
        transform = np.random.randint(0, 8)
        if transform & 1:
            board_2d = board_2d[:, ::-1, :]
        if transform & 2:
            board_2d = board_2d[:, :, ::-1]
        if transform & 4:
            board_2d = board_2d.transpose(0, 2, 1)
        return np.ascontiguousarray(board_2d)

    def _line_features(self, board, sides):
        n = board.shape[0]
        codes_us = np.zeros((n, NUM_LINES), dtype=np.uint32)
        codes_them = np.zeros((n, NUM_LINES), dtype=np.uint32)

        white_side = sides == 0
        black_side = ~white_side

        for sq in range(BOARD_SQ):
            cell = board[:, sq]
            white = cell == ord("1")
            black = cell == ord("2")
            arrow = cell == ord("3")

            piece_us = np.zeros(n, dtype=np.uint32)
            piece_them = np.zeros(n, dtype=np.uint32)
            piece_us[arrow] = 3
            piece_them[arrow] = 3

            piece_us[white & white_side] = 1
            piece_us[white & black_side] = 2
            piece_us[black & black_side] = 1
            piece_us[black & white_side] = 2

            piece_them[white & white_side] = 2
            piece_them[white & black_side] = 1
            piece_them[black & black_side] = 2
            piece_them[black & white_side] = 1

            if not piece_us.any() and not piece_them.any():
                continue

            for line, shift in SQ_LINES[sq]:
                codes_us[:, line] |= piece_us << np.uint32(shift)
                codes_them[:, line] |= piece_them << np.uint32(shift)

        line_ids = np.arange(NUM_LINES, dtype=np.uint32)
        hash_us = mix_line_hash(codes_us ^ LINE_SEED[None, :] ^ (line_ids[None, :] * np.uint32(0x9E3779B9)))
        hash_them = mix_line_hash(codes_them ^ LINE_SEED[None, :] ^ (line_ids[None, :] * np.uint32(0x9E3779B9)))

        idx_us = (hash_us & np.uint32(LINE_FEATURE_HASH - 1)).astype(np.int64)
        idx_them = (hash_them & np.uint32(LINE_FEATURE_HASH - 1)).astype(np.int64)
        sign_us = np.where((hash_us & np.uint32(0x80000000)) != 0, 1.0, -1.0).astype(np.float32)
        sign_them = np.where((hash_them & np.uint32(0x80000000)) != 0, 1.0, -1.0).astype(np.float32)
        return idx_us, sign_us, idx_them, sign_them

    def __call__(self, batch):
        groups = defaultdict(list)
        for path, idx in batch:
            groups[path].append(idx)

        chunks = []
        for path, indices in groups.items():
            if path not in self.handles:
                self.handles[path] = np.memmap(
                    path,
                    dtype=self.record_dtype,
                    mode="r",
                    offset=self.header_dtype.itemsize,
                )
            chunks.append(self.handles[path][indices])

        rec = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
        n = len(rec)
        sides = rec["side"]
        board_2d = np.ascontiguousarray(rec["board"]).view(np.uint8).reshape(n, BOARD_SIZE, BOARD_SIZE)

        if self.augment_d8:
            board_2d = self._apply_batch_d8(board_2d)

        board = board_2d.reshape(n, BOARD_SQ)

        us_arr = np.zeros((n, FEATURE_SIZE), dtype=np.float32)
        them_arr = np.zeros((n, FEATURE_SIZE), dtype=np.float32)

        white_mask = board == ord("1")
        black_mask = board == ord("2")
        arrow_mask = board == ord("3")
        white_side = sides == 0
        black_side = sides == 1

        us_arr[white_side, 0:100] = white_mask[white_side]
        them_arr[white_side, 100:200] = white_mask[white_side]
        us_arr[white_side, 100:200] = black_mask[white_side]
        them_arr[white_side, 0:100] = black_mask[white_side]

        us_arr[black_side, 0:100] = black_mask[black_side]
        them_arr[black_side, 100:200] = black_mask[black_side]
        us_arr[black_side, 100:200] = white_mask[black_side]
        them_arr[black_side, 0:100] = white_mask[black_side]

        arrows = arrow_mask.astype(np.float32)
        us_arr[:, 200:300] = arrows
        them_arr[:, 200:300] = arrows

        idx_us, sign_us, idx_them, sign_them = self._line_features(board, sides)
        target = ((rec["outcome"].astype(np.float32) + 1.0) / 2.0).reshape(-1, 1)
        score = rec["score"].astype(np.float32).reshape(-1, 1)
        classical = rec["classical"].astype(np.float32).reshape(-1, 1)
        global_features = rec["global"].astype(np.float32) / 127.0
        phase = np.clip(rec["phase_bucket"].astype(np.int64), 0, PHASE_BUCKETS - 1)

        return (
            torch.from_numpy(us_arr),
            torch.from_numpy(them_arr),
            torch.from_numpy(idx_us),
            torch.from_numpy(sign_us),
            torch.from_numpy(idx_them),
            torch.from_numpy(sign_them),
            torch.from_numpy(global_features),
            torch.from_numpy(phase),
            torch.from_numpy(classical),
            torch.from_numpy(target),
            torch.from_numpy(score),
        )
