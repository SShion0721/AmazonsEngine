# 亚马逊棋引擎 —— Stockfish架构学习指南

## 项目结构

```
AmazonsEngine/
├── types.h        ← 基础类型、常量、坐标工具（对应 SF: types.h）
├── rays.h/cpp     ← 预计算滑动攻击射线（对应 SF: magic bitboards）
├── position.h/cpp ← 棋盘状态、Zobrist哈希、走法执行（对应 SF: position.h）
├── movegen.h/cpp  ← 走法生成器（对应 SF: movegen.h）
├── evaluate.h/cpp ← 静态评估：领地+机动性（对应 SF: evaluate.cpp）
├── tt.h           ← 置换表（对应 SF: tt.h）
├── search.h/cpp   ← Negamax PVS搜索引擎（对应 SF: search.cpp）
└── main.cpp       ← UCI协议 + 主循环（对应 SF: uci.cpp + main.cpp）
```

## 编译

```bash
g++ -O2 -std=c++17 -o amazons.exe rays.cpp position.cpp movegen.cpp evaluate.cpp search.cpp main.cpp
```

## 运行命令

| 命令 | 说明 |
|---|---|
| `d` | 打印当前棋盘 |
| `eval` | 显示评估分数分解 |
| `perft <N>` | 计算深度N的节点数（测试走法生成器） |
| `go movetime 3000` | 搜索3秒并走棋 |
| `position startpos moves a4-a7/a5` | 加载局面并应用走法 |

## Stockfish概念对应表

| Stockfish概念 | 本引擎实现 | 说明 |
|---|---|---|
| Magic Bitboards | `RAYS[sq][dir]` | 预计算每格每方向的射线列表 |
| `Position::do_move` | `Position::do_move` | 增量更新：棋盘+Zobrist哈希 |
| `Position::undo_move` | `Position::undo_move` | 完全可逆（无吃子，只移动+放箭） |
| Transposition Table | `TranspositionTable` | Zobrist哈希+替换策略 |
| `search<PV>` | `Searcher::negamax` | Negamax PVS，全窗口/空窗口切换 |
| Late Move Reduction | `lmr_[depth][move_idx]` | log公式预计算，后排序走法降深 |
| Killer Moves | `killers_[depth][0..1]` | 同深度Beta截断的走法优先 |
| History Heuristic | `history_[color][from][to]` | 截断次数加权，指导走法排序 |
| Aspiration Window | 迭代加深的窄窗口 | 失败时扩大窗口重搜 |
| Reverse Futility | `static_eval - 60*depth >= beta` | 浅层节点提前剪枝 |

## 评估函数设计

### 领地评估（Voronoi BFS）
```
同时从双方所有亚马逊BFS扩展（王步）
每个格子归属距离更近的一方
分数 = 白方格数 - 黑方格数
```

### 机动性评估（Queen Reach）
```
统计每个亚马逊能立即到达的格数（后式滑动）
分数 = 白方总格数 - 黑方总格数  
```

### 最终分数
```
raw = 领地 × 10 + 机动性 × 2
返回值 = 当前行棋方视角（raw 或 -raw）
```

## 下一步：NNUE集成路线

```
1. 定义特征层（300个二值特征：棋子位置编码）
2. 用PyTorch训练：input(300) → L1(256) → L2(32) → output(1)
3. 自对弈生成训练数据（用当前Alpha-Beta搜索打标签）
4. 量化为INT8推理
5. 实现增量更新（每步走棋只更新3个格子对应的权重差量）
```

## 已知局限及改进方向

- **评估为0的bug**：初始局面双方对称，评估=0是正确的（双方完全对称）
- **Null Move**：亚马逊棋无法"过手"，已用反向Futility替代
- **Quiescence**：亚马逊棋无吃子，depth=0直接调用evaluate（合理）
- **走法排序**：可加入"箭矢封锁对方价值"的启发式分数
- **多线程**：UCI模式下可将搜索移出main线程（lazySMP并行化）
