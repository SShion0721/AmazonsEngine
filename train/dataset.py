import os
import numpy as np
import torch
from torch.utils.data import Dataset

FEATURE_SIZE = 300
BOARD_SIZE = 10

class AmazonsDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.record_dtype = np.dtype([
            ("side", np.int8),
            ("board", "S100"),
            ("outcome", np.int8),
            ("score", np.int16),
        ])
        
        self.data_path = data_path
        
        if not os.path.exists(data_path):
            print(f"Raw dataset path {data_path} not found.")
            self.num_samples = 0
            return

        # Just get the size
        stats = os.stat(data_path)
        self.num_samples = stats.st_size // self.record_dtype.itemsize
        
        self.data = None
        print(f"Mapped raw dataset: {data_path} ({self.num_samples} records, ~{stats.st_size / 1024**3:.2f} GB)")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Return path and index so RawCollate can fetch in bulk
        return self.data_path, idx

class RawCollate:
    def __init__(self, augment_d8=False):
        self.record_dtype = np.dtype([
            ("side", np.int8),
            ("board", "S100"),
            ("outcome", np.int8),
            ("score", np.int16),
        ])
        self.handles = {}
        self.augment_d8 = augment_d8

    def _apply_batch_d8(self, board_2d):
        # One vectorized symmetry per batch keeps throughput high while still
        # exposing every position to multiple equivalent orientations over time.
        transform = np.random.randint(0, 8)
        if transform & 1:
            board_2d = board_2d[:, ::-1, :]
        if transform & 2:
            board_2d = board_2d[:, :, ::-1]
        if transform & 4:
            board_2d = board_2d.transpose(0, 2, 1)
        return np.ascontiguousarray(board_2d)

    def __call__(self, batch):
        # batch is a list of (data_path, idx)
        # Group indices by path for efficient fancy indexing
        from collections import defaultdict
        groups = defaultdict(list)
        for path, idx in batch:
            groups[path].append(idx)
            
        all_recs = []
        for path, indices in groups.items():
            if path not in self.handles:
                self.handles[path] = np.memmap(path, dtype=self.record_dtype, mode='r')
            all_recs.append(self.handles[path][indices])
        
        # Merge all chunks into one batch structured array
        rec = np.concatenate(all_recs) if len(all_recs) > 1 else all_recs[0]
        num_samples = len(rec)

        # Vectorized extraction
        sides = rec["side"]
        outcomes = rec["outcome"]
        scores = rec["score"]
        board_2d = np.ascontiguousarray(rec["board"]).view(np.uint8).reshape(
            num_samples, BOARD_SIZE, BOARD_SIZE
        )

        if self.augment_d8:
            board_2d = self._apply_batch_d8(board_2d)

        board = board_2d.reshape(num_samples, FEATURE_SIZE // 3)

        # Feature transformation (Vectorized)
        us_arr = np.zeros((num_samples, FEATURE_SIZE), dtype=np.float32)
        them_arr = np.zeros((num_samples, FEATURE_SIZE), dtype=np.float32)

        white_mask = (board == ord('1'))
        black_mask = (board == ord('2'))
        arrow_mask = (board == ord('3'))

        white_side_mask = (sides == 0)
        black_side_mask = (sides == 1)

        us_arr[white_side_mask, 0:100] = white_mask[white_side_mask]
        them_arr[white_side_mask, 100:200] = white_mask[white_side_mask]
        us_arr[white_side_mask, 100:200] = black_mask[white_side_mask]
        them_arr[white_side_mask, 0:100] = black_mask[white_side_mask]

        us_arr[black_side_mask, 0:100] = black_mask[black_side_mask]
        them_arr[black_side_mask, 100:200] = black_mask[black_side_mask]
        us_arr[black_side_mask, 100:200] = white_mask[black_side_mask]
        them_arr[black_side_mask, 0:100] = white_mask[black_side_mask]

        arrows = arrow_mask.astype(np.float32)
        us_arr[:, 200:300] = arrows
        them_arr[:, 200:300] = arrows

        target_arr = ((outcomes.astype(np.float32) + 1.0) / 2.0).reshape(-1, 1)
        score_arr = scores.astype(np.float32).reshape(-1, 1)

        return torch.from_numpy(us_arr), torch.from_numpy(them_arr), torch.from_numpy(target_arr), torch.from_numpy(score_arr)
