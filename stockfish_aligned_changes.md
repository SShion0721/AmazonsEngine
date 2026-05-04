# Stockfish-Aligned NNUE Training Pipeline — Changes Summary

All 3 core fixes have been implemented, compiled, and validated end-to-end.

## ✅ Fix 1: Random Openings (Position Diversity)

**Problem**: Deterministic search = 200万盘完全相同的棋。

**Solution**: Each selfplay game now plays **4–8 random plies** before search begins.

| Detail | Implementation |
|--------|---------------|
| RNG | `std::mt19937` per thread, uniquely seeded via clock + Knuth hash |
| Plies | `4 + rng() % 5` (4 to 8 random moves) |
| Safety | Skips game if random opening reaches terminal state |
| Perf | Reusable `moves` vector (`reserve(2048)`) avoids heap churn |

**File**: [main.cpp](file:///e:/Desktop/AmazonsEngine/main.cpp) — `worker_func`

## ✅ Fix 2: Search Score Recording (Knowledge Distillation)

**Problem**: Only saving win/loss (`±1`) — too coarse for learning positional nuance.

**Solution**: Every position now records the **search score** from the evaluating side's perspective.

```diff
 #pragma pack(push, 1)
 struct GameRecord {
     int8_t side;
     char board[100];
     int8_t outcome;
+    int16_t score;     // search score (side-to-move perspective)
 };
 #pragma pack(pop)
```

| Component | Size Change |
|-----------|------------|
| `GameRecord` (C++) | 102 → **104 bytes** |
| `FeatureRecord` (converter) | 601 → **603 bytes** |

**Files modified**:
- [search.h](file:///e:/Desktop/AmazonsEngine/search.h) — `search()` gains `Score* out_score` parameter
- [search.cpp](file:///e:/Desktop/AmazonsEngine/search.cpp) — writes `best_score` before return
- [main.cpp](file:///e:/Desktop/AmazonsEngine/main.cpp) — records score per position, direct board copy (no `std::string`)
- [convert_selfplay.cpp](file:///e:/Desktop/AmazonsEngine/train/convert_selfplay.cpp) — passes score through

## ✅ Fix 3: Blended Loss Function (Stockfish-Style)

**Problem**: Pure BCE on win/loss — AlphaZero-style, can't learn "how much better".

**Solution**: `Target = λ·Outcome + (1−λ)·Sigmoid(Score / Scale)`

```python
score_win_prob = torch.sigmoid(b_score / score_scale)
blended_target = blend_lambda * b_target + (1.0 - blend_lambda) * score_win_prob
loss = criterion(pred, blended_target)
```

| Parameter | Default | CLI Flag | Purpose |
|-----------|---------|----------|---------|
| λ (lambda) | 0.5 | `--blend-lambda` | 0 = pure distillation, 1 = pure outcome |
| Scale | 400.0 | `--score-scale` | Sigmoid denominator for score → win prob |

**Files modified**:
- [dataset.py](file:///e:/Desktop/AmazonsEngine/train/dataset.py) — returns 4-tuple `(us, them, target, score)`
- [train.py](file:///e:/Desktop/AmazonsEngine/train/train.py) — blended loss computation + new CLI args

## Performance Optimizations

| Optimization | Impact |
|-------------|--------|
| Direct board copy (`'0' + board[sq]`) | Eliminates `std::string` allocation per move |
| `moves.reserve(2048)` reused across games | Avoids repeated heap allocation in hot loop |
| `GameRecord` built inline during play | No separate `game_history` → `records` conversion pass |
| Thread-local `std::mt19937` | No lock contention on RNG |
| Score clamping without `<algorithm>` | Zero extra header dependency |

## Usage

```bash
# 1. Generate 2M games (depth 4, 8 threads, 32MB hash/thread)
echo "selfplay 2000000 4 50 8 32" | .\build\amazons.exe

# 2. Train with blended loss (cold start: lambda=0 for pure distillation)
python train/train.py --blend-lambda 0.0 --epochs 100

# 3. Later: retrain with balanced blend
python train/train.py --blend-lambda 0.5 --epochs 200 --resume
```

> [!TIP]
> For cold-start training, use `--blend-lambda 0.0` (pure knowledge distillation from classical eval).
> Once the network has learned basic positional understanding, switch to `--blend-lambda 0.5` for self-play reinforcement.
