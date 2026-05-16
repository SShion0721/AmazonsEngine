# AmazonsEngine

10x10 亚马逊棋引擎。当前版本已经不是单纯 Stockfish-like AlphaBeta，而是：

- PEXT bitboard 走法生成
- cache-line cluster TT
- 分层 Classical2 评估
- AMZNUE2 / NNUE-v2 接口
- Match 模式 root MCTS/UCT + AB verify
- FastSelfplay / TeacherSafe / Match 三套搜索模式

## 构建

推荐使用 CMake Release：

```powershell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
```

Release 会启用：

- MSVC: `/O2 /DNDEBUG /GL /arch:AVX2`
- GCC/Clang: `-O3 -DNDEBUG -march=native -flto -funroll-loops`

## 常用命令 

| 命令 | 说明 |
|---|---|
| `uci` | 输出 UCI 选项 |
| `isready` | 同步 |
| `ucinewgame` | 清 TT / eval cache / history |
| `d` | 打印当前棋盘 |
| `eval` | 打印当前评估拆解 |
| `perft <N>` | 完整走法生成节点数 |
| `go depth <N>` | AlphaBeta 深度搜索 |
| `go movetime <ms>` | 时间搜索；Match 模式开中局会优先走 MCTS root |
| `selfplay <games> <depth> <ms> <gameThreads> <hashMB> <searchThreads> [nodes]` | 生成自博数据 |

## 搜索模式

### FastSelfplay

速度优先，用于快速生成局面分布。

推荐配置：

```text
setoption name Search Mode value FastSelfplay
setoption name Use NNUE value false
selfplay 20000000 4 1 15 1 1
```

该模式会启用快速评估、bounded movegen，并关闭强 move categories / MCTS。它追求吞吐，不适合当强 teacher。

### TeacherSafe

训练强标签优先。

推荐配置：

```text
setoption name Search Mode value TeacherSafe
setoption name Use NNUE value false
selfplay 100000 4 30 8 8 1
```

该模式关闭硬截断、NMP、LMP、MCTS，使用强 Classical2。速度慢是正常的。

### Match

实战默认模式。

```text
setoption name Search Mode value Match
go movetime 1000
```

开中局默认使用 root-only MCTS/UCT，后期或低时间/低深度回退到 AlphaBeta。MCTS 后会对前若干候选做浅层 AB verify。

## 当前评估体系

当前评估不是单一函数，而是分层路径：

```text
evaluate()
  -> evaluate_with_tier()
       -> Fast legacy eval
       -> Mixed selective eval
       -> Strong Classical2
       -> optional AMZNUE2 residual

MCTS root / rollout
  -> ChampionEvalFast
  -> Champion move prior
```

### 1. Fast legacy eval

用于 FastSelfplay、浅层非关键节点、NNUE/Strong Classical 都关闭时。

组成：

```text
legacy = TerritoryWeight * king-flood territory
       + MobilityWeight  * queen mobility
       + legacy partition bonus
```

特点：

- 很快，避免 queen/king distance 全量计算。
- 适合快速自博吞吐。
- 棋力不如 Strong Classical2。

### 2. Strong Classical2

这是当前主要强评估，结果为 side-to-move POV。

先计算双方距离场：

- `qdist[2][100]`: queen-distance，多源 BFS，扩展使用 `get_queen_attacks()`。
- `kdist[2][100]`: king-distance，bitboard wavefront 扩展。

然后得到以下白方视角子项：

| 子项 | 含义 |
|---|---|
| `t1` | queen-distance territory：空格离哪方 queen 距离更近 |
| `t2` | king-distance territory：空格离哪方 king 距离更近 |
| `p1` | queen-distance closeness：越近权重越高 |
| `p2` | king-distance closeness / reachability 修正 |
| `m` | per-amazon mobility/enclosure：局部机动性和被困惩罚 |
| `partition` | 连通块/分区评估 |
| `mobility` | PEXT queen mobility 总差 |

阶段权重按箭数变化：

```text
phase 0: arrows <= 14
phase 1: arrows <= 25
phase 2: arrows <= 40
phase 3: arrows <= 55
phase 4: arrows <= 63
phase 5: arrows >  63
```

越到后期，`t2`、`m`、`partition` 权重越高；开局更偏 `t1`。

### 3. Partition fast-track

当箭数较多时，会尝试检查棋盘是否已经完全分区：

```text
if no connected component contains both white and black amazons:
    eval = large side-relative component-space score
```

这避免后期在已经分开的局面里继续做大量无意义搜索。

小单边区域还会尝试 exact fill solver：

```text
if component contains only one side
and empty_count <= 14:
    exact = local maximum-fill search with memo
else:
    fallback = simple component estimate
```

这个 solver 只在小 component 上工作。它会把局部区域内的 amazon 位置和占用位作为 memo key，精确搜索“这一块区域最终还能走几步”。如果局部分支爆炸超过预算，会自动回退到原来的估值，避免拖慢中盘。

### 4. Eval cache

Strong Classical2 的距离场和 global features 不在 `do_move()` 里维护，而是在 evaluate cache miss 时计算。

当前 cache 是：

```text
thread_local direct-mapped eval cache
size = 1 << 15
key  = position zobrist
data = EvalInfo { breakdown, global[32], phase_bucket }
```

好处：

- 多线程 selfplay 不抢同一个全局 eval cache。
- 关闭 Strong Classical / NNUE 时不会误触发重评估。

### 5. AMZNUE2 / Fusion NNUE

当前旧 300-only NNUE 已废弃。加载器只接受 AMZNUE2 格式。

NNUE active 条件：

```text
Use NNUE = true
and AMZNUE2 weights loaded
```

推理路径：

```text
classical = Classical2 or legacy
nnue_eval = AMZNUE2(base_acc, line_acc, global[32], phase_bucket)

if Use Pure NNUE:
    return nnue_eval
else if Use Residual NNUE:
    return classical + nnue_eval
else:
    return classical
```

增量 accumulator：

```text
base[2][512]
line[2][64]
line_code[2][58]
```

`do_move()` 只在 NNUE active 时复制/更新 accumulator，避免 `Use NNUE=false` 的 selfplay 被拖慢。

### 6. ChampionEvalFast

这是给 MCTS root / rollout / prior 用的轻量冠军式评估，不直接替代 Classical2。

组成：

```text
ChampionEvalFast =
    phase_weighted T1 queen territory
  + phase_weighted T2 king territory
  + local mobility / enclosure
```

它的目标不是比 Classical2 更准，而是：

- 给 MCTS rollout leaf 一个稳定、便宜的局面值。
- 给 root candidate 一个“空间切割/封锁”风格 prior。
- 避免 MCTS 变成纯随机 playout。

Move prior 会考虑：

- 移动后自身 mobility 是否提升。
- 箭是否邻近敌方 amazon。
- 箭是否减少敌方 queen mobility。
- 箭是否打在低邻居数 chokepoint。
- 是否自困。

## evaluate_search 的选择逻辑

搜索中不是每个节点都强评估：

```text
if Strong Classical disabled and NNUE inactive:
    fast legacy eval

if Eval Tier = Strong or NNUE active:
    Strong Classical2 / Fusion

if Eval Tier = Fast:
    fast legacy eval

if Eval Tier = Mixed:
    root / PV / depth >= 3 / arrows >= 56 -> Strong
    otherwise first use Fast
    if Fast score near alpha/beta window -> Strong re-eval
```

这个设计是为了避免每个叶子都算 queen/king distance，同时保留关键节点的评估质量。

## MCTS/AB 混合搜索

Match 模式触发 MCTS 条件：

```text
Search Mode = Match
Use MCTS Root = true
not TeacherSafe
node limit disabled
max_depth >= 8
arrows < 56
remaining time >= MCTS Min Time
```

MCTS 设置：

```text
opening:
    time share = max(option, 85%)
    root initial width = 192
    root max candidates = 512
    rollout depth = 6
    AB verify topN <= 4

midgame:
    time share = option default 75%
    root initial width = 160
    root max candidates = 448
    rollout depth = 6

late pre-partition:
    time share <= 60%
    root initial width = 96
    root max candidates = 256
    rollout depth = 4
    AB verify topN >= 8

UCT C = 0.45
default MCTS Time Share = 75%
default AB Verify TopN = 6
```

MCTS 最终优先选择 root child 访问数最多的走法，访问数接近时用 mean value + prior 破平。

注意：`MCTS Time Share` 和 `AB Verify TopN` 是基础参数。Match 模式会根据箭数和剩余时间做自适应调整：开局更重广度，中后期更重 AB 校验。

## MovePicker / history

当前 MovePicker 使用固定数组，不在热路径分配 vector。

排序信息：

```text
history_from_to
history_to_arrow
history_from_arrow
butterfly_from_to
butterfly_to_arrow
butterfly_from_arrow
category score
```

相对 history：

```text
relative = history + history * 1024 / (butterfly + 64)
```

这样可以奖励“搜索次数少但 cutoff 率高”的远距离封锁箭，避免它们被常见平庸走法淹没。

## Soft Tail Search

Match/TeacherSafe 不应该硬丢弃 tail moves。

当前策略：

```text
top-K:
    正常排序、正常深度搜索

tail:
    仍保留
    强 LMR
    zero-window
    fail-high 后 full-depth re-search
```

FastSelfplay 可以继续使用 bounded movegen 和硬截断，因为目标是吞吐。

## 当前验证基线

最近一次验证：

```text
verify_refactor: accumulator, movegen, pseudo-legal, undo checks passed
perft 1 = 2176
perft 2 = 4307152
go depth 3: returns legal bestmove
go movetime 1000: MCTS returns legal bestmove
```

microbench 参考：

```text
movegen                 ~2.3 us/call
movepicker              ~36 us/call
evaluate fast           ~0.11 us/call
strong eval cache hit   ~0.02 us/call
strong eval cache miss  ~8.5 us/call
NNUE forward            ~2 us/call
```

## 训练数据建议

快速局面分布：

```text
setoption name Search Mode value FastSelfplay
setoption name Use NNUE value false
selfplay 20000000 4 1 15 1 1
```

强 teacher 标签：

```text
setoption name Search Mode value TeacherSafe
setoption name Use NNUE value false
selfplay 100000 4 30 8 8 1
```

实战风格数据：

```text
setoption name Search Mode value Match
setoption name Use NNUE value false
selfplay 100000 8 1000 1 64 1
```

训练 AMZNUE2 时，生成 teacher 数据通常应关闭 NNUE，避免旧网络或未收敛网络污染标签。

## 配置与棋力提升建议

为了获得最强的对弈表现，建议根据硬件和需求调整以下配置：

### 1. 核心搜索模式 (Search Mode)
- **Match (推荐)**：这是最强的对弈模式。它结合了 MCTS 的全局视野和 Alpha-Beta 的战术精确度。建议在正式比赛中使用。
- **FastSelfplay**：专为快速生成训练数据设计，牺牲了一部分深度以换取极高的吞吐量。
- **TeacherSafe**：用于生成更高质量的训练标签。

### 2. MCTS 关键参数
- **Use MCTS Root (开启)**：在亚马逊棋这种高分支、重空间的博弈中，MCTS 能够更好地识别“领土封锁”趋势。这是引擎能够赢下许多复杂残局的关键配置。
- **MCTS Time Share (默认 75%)**：控制 MCTS 分配的时间比例。增加此值可提升战略眼光，更早地发现获胜路径。

### 3. 评估系统 (Evaluation)
- **Use NNUE (推荐开启)**：启用 AMZNUE2 神经网络。这是提升棋力的单次最大增益项。开启后，引擎将从单纯的几何评估进化到对细微格局的深度理解。
- **Eval Tier (推荐 Mixed/Strong)**：在混合模式下，关键节点使用强评估，普通节点使用快速评估，在速度和精度间达到最佳平衡。

### 4. 硬件资源利用
- **Threads**：建议设置为 CPU 的逻辑核心数。本引擎支持多线程并行 MCTS 和共享 TT 表，核心数越多，每秒搜索的节点数 (nps) 越高，棋力越强。
- **Hash**：建议设置为 1024MB 或更高。更大的哈希表能显著提升中后期搜索的稳定性，减少重复计算。

### 5. 常见致胜组合
- **最强对弈配置**：`Search Mode = Match` + `Use MCTS Root = true` + `Use NNUE = true` + `Threads = [CPU 最大核心数]` + `Hash = 1024MB+`。


