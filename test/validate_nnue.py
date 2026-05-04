#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import queue
import random
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional


WHITE = 0
BLACK = 1
BOARD_SIZE = 10
MAX_GAME_PLIES = 200
DIRS = (
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, 1),
    (-1, -1),
    (1, -1),
    (-1, 1),
)


def sq_to_str(square: int) -> str:
    return f"{chr(ord('a') + (square % BOARD_SIZE))}{square // BOARD_SIZE + 1}"


def str_to_sq(text: str) -> int:
    if len(text) < 2:
        raise ValueError(f"Invalid square: {text}")
    file_idx = ord(text[0]) - ord("a")
    rank_idx = int(text[1:]) - 1
    if not (0 <= file_idx < BOARD_SIZE and 0 <= rank_idx < BOARD_SIZE):
        raise ValueError(f"Invalid square: {text}")
    return rank_idx * BOARD_SIZE + file_idx


def move_to_str(move: tuple[int, int, int]) -> str:
    return f"{sq_to_str(move[0])}-{sq_to_str(move[1])}/{sq_to_str(move[2])}"


def parse_move(text: str) -> tuple[int, int, int]:
    dash = text.find("-")
    slash = text.find("/")
    if dash < 0 or slash < 0 or dash > slash:
        raise ValueError(f"Invalid move: {text}")
    return (
        str_to_sq(text[:dash]),
        str_to_sq(text[dash + 1:slash]),
        str_to_sq(text[slash + 1:]),
    )


def normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def score_to_elo(score: float) -> float:
    score = min(max(score, 1e-9), 1.0 - 1e-9)
    return -400.0 * math.log10((1.0 / score) - 1.0)


def percentile_to_z(confidence: float) -> float:
    # Supported common values without extra dependencies.
    lookup = {
        0.80: 1.2815515655446004,
        0.90: 1.6448536269514722,
        0.95: 1.959963984540054,
        0.98: 2.3263478740408408,
        0.99: 2.5758293035489004,
    }
    rounded = round(confidence, 2)
    if rounded in lookup:
        return lookup[rounded]
    # Fallback to 95% if the caller picks an uncommon value.
    return lookup[0.95]


@dataclass
class OpeningBoard:
    board: list[int]
    amazons: list[list[int]]
    side_to_move: int

    @classmethod
    def startpos(cls) -> "OpeningBoard":
        board = [0] * 100
        white = [30, 3, 6, 39]   # a4 d1 g1 j4
        black = [60, 93, 96, 69] # a7 d10 g10 j7
        for sq in white:
            board[sq] = 1
        for sq in black:
            board[sq] = 2
        return cls(board=board, amazons=[white[:], black[:]], side_to_move=WHITE)

    def _ray_moves(
        self,
        start_sq: int,
        empty_override: Optional[int] = None,
        occupied_override: Optional[int] = None,
    ) -> list[int]:
        targets: list[int] = []
        start_row = start_sq // BOARD_SIZE
        start_col = start_sq % BOARD_SIZE

        for dr, dc in DIRS:
            row = start_row + dr
            col = start_col + dc
            while 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
                sq = row * BOARD_SIZE + col
                occupied = self.board[sq] != 0
                if empty_override is not None and sq == empty_override:
                    occupied = False
                if occupied_override is not None and sq == occupied_override:
                    occupied = True
                if occupied:
                    break
                targets.append(sq)
                row += dr
                col += dc
        return targets

    def legal_moves(self) -> list[tuple[int, int, int]]:
        moves: list[tuple[int, int, int]] = []
        piece_value = 1 if self.side_to_move == WHITE else 2
        for from_sq in self.amazons[self.side_to_move]:
            if self.board[from_sq] != piece_value:
                continue
            for to_sq in self._ray_moves(from_sq):
                for arrow_sq in self._ray_moves(to_sq, empty_override=from_sq, occupied_override=to_sq):
                    moves.append((from_sq, to_sq, arrow_sq))
        return moves

    def play(self, move: tuple[int, int, int]) -> None:
        from_sq, to_sq, arrow_sq = move
        piece_value = 1 if self.side_to_move == WHITE else 2
        if self.board[from_sq] != piece_value:
            raise ValueError(f"No side-to-move amazon on {sq_to_str(from_sq)}")
        if self.board[to_sq] != 0 or self.board[arrow_sq] != 0 and arrow_sq != from_sq:
            raise ValueError("Move lands on an occupied square")

        self.board[from_sq] = 0
        self.board[to_sq] = piece_value
        self.board[arrow_sq] = 3

        amazons = self.amazons[self.side_to_move]
        idx = amazons.index(from_sq)
        amazons[idx] = to_sq
        self.side_to_move ^= 1


def random_opening(plies: int, rng: random.Random, retries: int = 100) -> list[str]:
    for _ in range(retries):
        board = OpeningBoard.startpos()
        opening: list[str] = []
        ok = True
        for _ply in range(plies):
            moves = board.legal_moves()
            if not moves:
                ok = False
                break
            move = rng.choice(moves)
            opening.append(move_to_str(move))
            board.play(move)
        if ok:
            return opening
    raise RuntimeError(f"Could not generate a legal random opening after {retries} retries")


class EngineProtocolError(RuntimeError):
    pass


@dataclass
class EngineConfig:
    engine_path: Path
    eval_file: Optional[Path]
    use_nnue: bool
    threads: int
    hash_mb: int
    move_overhead: int
    extra_options: list[tuple[str, str]] = field(default_factory=list)


class EngineProcess:
    def __init__(self, label: str, config: EngineConfig):
        self.label = label
        self.config = config
        self.proc: Optional[subprocess.Popen[str]] = None
        self.reader: Optional[threading.Thread] = None
        self.lines: "queue.Queue[Optional[str]]" = queue.Queue()

    def start(self) -> None:
        if self.proc is not None:
            return

        self.proc = subprocess.Popen(
            [str(self.config.engine_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="ignore",
            bufsize=1,
            cwd=str(self.config.engine_path.parent),
        )
        self.reader = threading.Thread(target=self._reader_loop, daemon=True)
        self.reader.start()
        self._drain_output()
        self._uci_handshake()
        self.configure()

    def _reader_loop(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        for raw_line in self.proc.stdout:
            self.lines.put(raw_line.rstrip("\r\n"))
        self.lines.put(None)

    def _send(self, command: str) -> None:
        if self.proc is None or self.proc.stdin is None:
            raise EngineProtocolError(f"{self.label}: engine is not running")
        try:
            self.proc.stdin.write(command + "\n")
            self.proc.stdin.flush()
        except OSError as exc:
            raise EngineProtocolError(f"{self.label}: failed to send command {command!r}: {exc}") from exc

    def _read_until(self, predicate, timeout: float) -> list[str]:
        deadline = time.time() + timeout
        collected: list[str] = []
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                raise EngineProtocolError(
                    f"{self.label}: timed out after {timeout:.1f}s while waiting for engine output"
                )
            try:
                line = self.lines.get(timeout=remaining)
            except queue.Empty as exc:
                raise EngineProtocolError(
                    f"{self.label}: timed out after {timeout:.1f}s while waiting for engine output"
                ) from exc
            if line is None:
                rc = self.proc.poll() if self.proc is not None else "unknown"
                raise EngineProtocolError(f"{self.label}: engine terminated unexpectedly (exit={rc})")
            collected.append(line)
            if predicate(line):
                return collected

    def _drain_output(self) -> None:
        while True:
            try:
                self.lines.get_nowait()
            except queue.Empty:
                return

    def _uci_handshake(self) -> None:
        self._send("uci")
        self._read_until(lambda line: line == "uciok", timeout=5.0)
        self.is_ready()

    def is_ready(self) -> None:
        self._send("isready")
        self._read_until(lambda line: line == "readyok", timeout=10.0)

    def set_option(self, name: str, value: Optional[str]) -> list[str]:
        self._drain_output()
        if value is None:
            self._send(f"setoption name {name}")
        else:
            self._send(f"setoption name {name} value {value}")
        self._send("isready")
        return self._read_until(lambda line: line == "readyok", timeout=10.0)

    def configure(self) -> None:
        self.set_option("Threads", str(self.config.threads))
        self.set_option("Hash", str(self.config.hash_mb))
        self.set_option("Move Overhead", str(self.config.move_overhead))

        use_nnue_value = "true" if self.config.use_nnue else "false"
        lines = self.set_option("Use NNUE", use_nnue_value)
        info = "\n".join(line for line in lines if line.startswith("info string"))
        if self.config.use_nnue:
            if "Use NNUE enabled" not in info:
                raise EngineProtocolError(
                    f"{self.label}: expected Use NNUE=true acknowledgement, got:\n{info}"
                )
        else:
            if "Use NNUE disabled" not in info:
                raise EngineProtocolError(
                    f"{self.label}: expected Use NNUE=false acknowledgement, got:\n{info}"
                )

        if self.config.eval_file is not None:
            eval_target = str(self.config.eval_file.resolve())
            lines = self.set_option("EvalFile", eval_target)
            info = "\n".join(line for line in lines if line.startswith("info string"))
            if "NNUE weights loaded from" not in info:
                raise EngineProtocolError(
                    f"{self.label}: expected NNUE load success for {self.config.eval_file}, got:\n{info}"
                )
        elif self.config.use_nnue:
            raise EngineProtocolError(
                f"{self.label}: Use NNUE=true requires an eval file, but none was configured"
            )

        for name, value in self.config.extra_options:
            self.set_option(name, value)

        mode = self.query_eval_mode()
        expected_nnue = self.config.use_nnue and self.config.eval_file is not None
        if expected_nnue and "NNUE" not in mode:
            raise EngineProtocolError(
                f"{self.label}: expected an NNUE eval mode, got {mode}"
            )
        if not expected_nnue and mode != "Classical":
            raise EngineProtocolError(
                f"{self.label}: expected eval mode Classical, got {mode}"
            )

    def query_eval_mode(self) -> str:
        self._drain_output()
        self._send("position startpos")
        self._send("eval")
        lines = self._read_until(lambda line: line.startswith("Total     :"), timeout=5.0)
        for line in lines:
            if line.startswith("Mode      :"):
                return line.split(":", 1)[1].strip()
        raise EngineProtocolError(f"{self.label}: eval output did not include a Mode line")

    def new_game(self) -> None:
        self._drain_output()
        self._send("ucinewgame")
        self.is_ready()

    def bestmove(self, moves: list[str], movetime_ms: int, timeout_s: float) -> tuple[str, list[str]]:
        self._drain_output()
        if moves:
            self._send("position startpos moves " + " ".join(moves))
        else:
            self._send("position startpos")
        self._send(f"go movetime {movetime_ms}")
        lines = self._read_until(lambda line: line.startswith("bestmove"), timeout=timeout_s)
        bestmove = "0000"
        info_lines: list[str] = []
        for line in lines:
            if line.startswith("bestmove"):
                parts = line.split()
                bestmove = parts[1] if len(parts) >= 2 else "0000"
            elif line.startswith("info"):
                info_lines.append(line)
        return bestmove, info_lines

    def close(self) -> None:
        if self.proc is None:
            return
        try:
            if self.proc.poll() is None:
                self._send("quit")
                self.proc.wait(timeout=2.0)
        except Exception:
            self.proc.kill()
            self.proc.wait(timeout=2.0)
        finally:
            self.proc = None


@dataclass
class GameResult:
    score: float
    winner: str
    reason: str
    plies_played: int
    opening_moves: list[str]
    final_moves: list[str]
    failed_engine: str = ""
    failed_side: str = ""
    reproduce_position: str = ""
    info_lines: list[str] = field(default_factory=list)


def position_command_from_moves(moves: list[str]) -> str:
    if moves:
        return "position startpos moves " + " ".join(moves)
    return "position startpos"


def print_failure_debug(prefix: str, result: GameResult) -> None:
    print(f"  {prefix} failure detail:")
    print(f"    winner      : {result.winner}")
    print(f"    reason      : {result.reason}")
    print(f"    failed side : {result.failed_side or 'unknown'}")
    print(f"    failed eng  : {result.failed_engine or 'unknown'}")
    print(f"    opening     : {' '.join(result.opening_moves) if result.opening_moves else '(none)'}")
    print(f"    full moves  : {' '.join(result.final_moves) if result.final_moves else '(none)'}")
    print(f"    reproduce   : {result.reproduce_position or position_command_from_moves(result.final_moves)}")
    if result.info_lines:
        print("    last info   :")
        for line in result.info_lines[-5:]:
            print(f"      {line}")


@dataclass
class MatchStats:
    pair_scores: list[float] = field(default_factory=list)
    wins: int = 0
    losses: int = 0
    draws: int = 0
    pentanomial: dict[str, int] = field(default_factory=lambda: {
        "LL": 0,
        "LD/DL": 0,
        "WL/DD": 0,
        "WD/DW": 0,
        "WW": 0,
    })

    def add_pair(self, first: GameResult, second: GameResult) -> None:
        for result in (first, second):
            if result.score == 1.0:
                self.wins += 1
            elif result.score == 0.0:
                self.losses += 1
            else:
                self.draws += 1

        total = first.score + second.score
        self.pair_scores.append(total)

        if math.isclose(total, 0.0):
            self.pentanomial["LL"] += 1
        elif math.isclose(total, 0.5):
            self.pentanomial["LD/DL"] += 1
        elif math.isclose(total, 1.0):
            self.pentanomial["WL/DD"] += 1
        elif math.isclose(total, 1.5):
            self.pentanomial["WD/DW"] += 1
        elif math.isclose(total, 2.0):
            self.pentanomial["WW"] += 1
        else:
            raise ValueError(f"Unexpected pair score: {total}")

    @property
    def games(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def pairs(self) -> int:
        return len(self.pair_scores)

    @property
    def score_rate(self) -> float:
        games = self.games
        if games == 0:
            return 0.5
        return (self.wins + 0.5 * self.draws) / games

    def confidence_interval(self, confidence: float) -> tuple[float, float]:
        if not self.pair_scores:
            return 0.5, 0.5
        if len(self.pair_scores) == 1:
            mean_game = self.pair_scores[0] / 2.0
            return mean_game, mean_game

        mean_pair = sum(self.pair_scores) / len(self.pair_scores)
        variance_pair = sum((value - mean_pair) ** 2 for value in self.pair_scores) / (len(self.pair_scores) - 1)
        se_game = math.sqrt(variance_pair / len(self.pair_scores)) / 2.0
        z = percentile_to_z(confidence)
        mean_game = mean_pair / 2.0
        low = max(0.0, mean_game - z * se_game)
        high = min(1.0, mean_game + z * se_game)
        return low, high

    def los(self) -> float:
        if not self.pair_scores:
            return 0.5
        if len(self.pair_scores) == 1:
            return 1.0 if self.score_rate > 0.5 else 0.0 if self.score_rate < 0.5 else 0.5

        mean_pair = sum(self.pair_scores) / len(self.pair_scores)
        variance_pair = sum((value - mean_pair) ** 2 for value in self.pair_scores) / (len(self.pair_scores) - 1)
        se_game = math.sqrt(variance_pair / len(self.pair_scores)) / 2.0
        if se_game == 0.0:
            return 1.0 if self.score_rate > 0.5 else 0.0 if self.score_rate < 0.5 else 0.5
        z = (self.score_rate - 0.5) / se_game
        return normal_cdf(z)


def parse_name_value(values: Iterable[str]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Option must be NAME=VALUE, got: {value}")
        name, raw = value.split("=", 1)
        name = name.strip()
        raw = raw.strip()
        if not name:
            raise ValueError(f"Option name is empty in: {value}")
        pairs.append((name, raw))
    return pairs


def load_openings_file(path: Path) -> list[list[str]]:
    openings: list[list[str]] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        moves = line.split()
        for move in moves:
            parse_move(move)
        openings.append(moves)
    if not openings:
        raise ValueError(f"No openings found in {path}")
    return openings


def play_game(
    opening_moves: list[str],
    candidate_as_white: bool,
    candidate: EngineProcess,
    baseline: EngineProcess,
    movetime_ms: int,
    timeout_s: float,
) -> GameResult:
    candidate_color = WHITE if candidate_as_white else BLACK
    history = list(opening_moves)
    side_to_move = WHITE if len(history) % 2 == 0 else BLACK
    board = OpeningBoard.startpos()
    for move_text in opening_moves:
        board.play(parse_move(move_text))

    candidate.new_game()
    baseline.new_game()

    for _ply in range(len(history), MAX_GAME_PLIES):
        engine = candidate if side_to_move == candidate_color else baseline
        bestmove, _info = engine.bestmove(history, movetime_ms=movetime_ms, timeout_s=timeout_s)
        if bestmove == "0000":
            candidate_wins = side_to_move != candidate_color
            remaining_legal = board.legal_moves()
            reason = "no legal move" if not remaining_legal else f"false 0000 ({len(remaining_legal)} legal moves exist)"
            return GameResult(
                score=1.0 if candidate_wins else 0.0,
                winner="candidate" if candidate_wins else "baseline",
                reason=reason,
                plies_played=len(history),
                opening_moves=list(opening_moves),
                final_moves=history,
                failed_engine=engine.label,
                failed_side="white" if side_to_move == WHITE else "black",
                reproduce_position=position_command_from_moves(history),
                info_lines=_info,
            )

        parse_move(bestmove)
        history.append(bestmove)
        board.play(parse_move(bestmove))
        side_to_move ^= 1

    return GameResult(
        score=0.5,
        winner="draw",
        reason=f"hit safety limit ({MAX_GAME_PLIES} plies)",
        plies_played=len(history),
        opening_moves=list(opening_moves),
        final_moves=history,
    )


def format_summary(
    stats: MatchStats,
    confidence: float,
    candidate_name: str,
    baseline_name: str,
) -> str:
    score = stats.score_rate
    low_score, high_score = stats.confidence_interval(confidence)
    elo = score_to_elo(score)
    low_elo = score_to_elo(low_score)
    high_elo = score_to_elo(high_score)
    los = stats.los() * 100.0

    lines = [
        f"Games: {stats.games} ({stats.pairs} paired openings)",
        f"Score: {candidate_name} {stats.wins} - {stats.losses} - {stats.draws} {baseline_name}",
        f"Win rate: {score * 100.0:.2f}%",
        f"Elo estimate: {elo:+.2f}  [{low_elo:+.2f}, {high_elo:+.2f}]  ({int(confidence * 100)}% CI)",
        f"LOS: {los:.2f}%",
        "Pentanomial: "
        + ", ".join(f"{key}={value}" for key, value in stats.pentanomial.items()),
    ]
    return "\n".join(lines)


def verdict_from_stats(stats: MatchStats, confidence: float, min_pairs: int) -> str:
    if stats.pairs < min_pairs:
        return "INCONCLUSIVE"
    low_score, high_score = stats.confidence_interval(confidence)
    low_elo = score_to_elo(low_score)
    high_elo = score_to_elo(high_score)
    if low_elo > 0.0:
        return "PASS"
    if high_elo < 0.0:
        return "FAIL"
    return "INCONCLUSIVE"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Stockfish-style paired-opening validation matches for AmazonsEngine.\n"
            "By default the baseline is classical eval only. Pass --baseline-eval-file "
            "later to compare against the current best NNUE."
        )
    )
    repo_root = Path(__file__).resolve().parent.parent
    default_engine = repo_root / "build" / "amazons.exe"

    parser.add_argument("--engine", type=Path, default=default_engine, help="Path to amazons.exe")
    parser.add_argument("--candidate-eval-file", type=Path, required=True, help="Candidate NNUE file to test")
    parser.add_argument(
        "--baseline-eval-file",
        type=Path,
        default=None,
        help="Optional baseline NNUE. Omit to use classical eval only.",
    )
    parser.add_argument(
        "--candidate-use-nnue",
        choices=["auto", "true", "false"],
        default="auto",
        help="Whether the candidate side should search with NNUE enabled (default: auto => true when eval file is present).",
    )
    parser.add_argument(
        "--baseline-use-nnue",
        choices=["auto", "true", "false"],
        default="auto",
        help="Whether the baseline side should search with NNUE enabled (default: auto => true only when baseline eval file is present).",
    )
    parser.add_argument("--games", type=int, default=40, help="Total games to play (rounded up to an even number)")
    parser.add_argument("--movetime", type=int, default=100, help="Per-move think time in ms")
    parser.add_argument("--threads", type=int, default=1, help="Threads per engine")
    parser.add_argument("--hash", type=int, default=64, help="Hash MB per engine")
    parser.add_argument("--move-overhead", type=int, default=10, help="Move overhead passed to the engine")
    parser.add_argument("--opening-plies", type=int, default=4, help="Random opening plies before the match")
    parser.add_argument("--openings-file", type=Path, default=None, help="Optional text file with one opening line per row")
    parser.add_argument("--seed", type=int, default=20260503, help="Random seed for generated openings")
    parser.add_argument("--timeout", type=float, default=15.0, help="Hard timeout per move in seconds")
    parser.add_argument("--confidence", type=float, default=0.95, help="Confidence level for Elo interval")
    parser.add_argument("--min-pairs", type=int, default=8, help="Minimum paired openings before PASS/FAIL is allowed")
    parser.add_argument("--report-every", type=int, default=5, help="Print an intermediate summary every N pairs")
    parser.add_argument("--early-stop", action="store_true", help="Stop as soon as PASS or FAIL becomes decisive")
    parser.add_argument("--candidate-name", default="candidate", help="Label used in match reports")
    parser.add_argument("--baseline-name", default=None, help="Optional label for the baseline side")
    parser.add_argument(
        "--candidate-option",
        action="append",
        default=[],
        help="Extra UCI option for candidate in NAME=VALUE form (repeatable)",
    )
    parser.add_argument(
        "--baseline-option",
        action="append",
        default=[],
        help="Extra UCI option for baseline in NAME=VALUE form (repeatable)",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.games <= 0:
        parser.error("--games must be positive")
    if args.movetime <= 0:
        parser.error("--movetime must be positive")
    if args.threads <= 0:
        parser.error("--threads must be positive")
    if args.hash <= 0:
        parser.error("--hash must be positive")
    if args.opening_plies < 0:
        parser.error("--opening-plies cannot be negative")
    if not (0.5 <= args.confidence < 1.0):
        parser.error("--confidence must be in [0.5, 1.0)")

    engine_path = args.engine.resolve()
    candidate_eval = args.candidate_eval_file.resolve()
    baseline_eval = args.baseline_eval_file.resolve() if args.baseline_eval_file else None

    if not engine_path.exists():
        parser.error(f"Engine not found: {engine_path}")
    if not candidate_eval.exists():
        parser.error(f"Candidate eval file not found: {candidate_eval}")
    if baseline_eval is not None and not baseline_eval.exists():
        parser.error(f"Baseline eval file not found: {baseline_eval}")

    candidate_use_nnue = (
        True if args.candidate_use_nnue == "auto" else args.candidate_use_nnue == "true"
    )
    baseline_use_nnue = (
        baseline_eval is not None if args.baseline_use_nnue == "auto" else args.baseline_use_nnue == "true"
    )

    if candidate_use_nnue and candidate_eval is None:
        parser.error("Candidate NNUE mode requires --candidate-eval-file")
    if baseline_use_nnue and baseline_eval is None:
        parser.error("Baseline NNUE mode requires --baseline-eval-file")

    total_games = args.games if args.games % 2 == 0 else args.games + 1
    total_pairs = total_games // 2

    candidate_options = parse_name_value(args.candidate_option)
    baseline_options = parse_name_value(args.baseline_option)

    openings_suite: Optional[list[list[str]]] = None
    if args.openings_file is not None:
        openings_suite = load_openings_file(args.openings_file.resolve())

    baseline_name = args.baseline_name
    if baseline_name is None:
        baseline_name = baseline_eval.stem if baseline_eval is not None else "classical"

    candidate_cfg = EngineConfig(
        engine_path=engine_path,
        eval_file=candidate_eval,
        use_nnue=candidate_use_nnue,
        threads=args.threads,
        hash_mb=args.hash,
        move_overhead=args.move_overhead,
        extra_options=candidate_options,
    )
    baseline_cfg = EngineConfig(
        engine_path=engine_path,
        eval_file=baseline_eval,
        use_nnue=baseline_use_nnue,
        threads=args.threads,
        hash_mb=args.hash,
        move_overhead=args.move_overhead,
        extra_options=baseline_options,
    )

    rng = random.Random(args.seed)
    candidate_engine = EngineProcess(args.candidate_name, candidate_cfg)
    baseline_engine = EngineProcess(baseline_name, baseline_cfg)
    stats = MatchStats()

    print(f"Engine         : {engine_path}")
    print(f"Candidate eval : {candidate_eval}")
    print(f"Candidate mode : {'NNUE' if candidate_use_nnue else 'Classical'}")
    print(f"Baseline eval  : {baseline_eval if baseline_eval is not None else '(none)'}")
    print(f"Baseline mode  : {'NNUE' if baseline_use_nnue else 'Classical'}")
    print(f"Pairs          : {total_pairs}")
    print(f"Movetime       : {args.movetime} ms")
    print(f"Seed           : {args.seed}")
    if openings_suite is not None:
        print(f"Openings file  : {args.openings_file.resolve()} ({len(openings_suite)} lines)")
    else:
        print(f"Opening plies  : {args.opening_plies} random plies")
    print()

    try:
        candidate_engine.start()
        baseline_engine.start()

        for pair_index in range(total_pairs):
            if openings_suite is not None:
                opening = list(openings_suite[pair_index % len(openings_suite)])
            else:
                opening = random_opening(args.opening_plies, rng)

            first = play_game(
                opening_moves=opening,
                candidate_as_white=True,
                candidate=candidate_engine,
                baseline=baseline_engine,
                movetime_ms=args.movetime,
                timeout_s=args.timeout,
            )
            second = play_game(
                opening_moves=opening,
                candidate_as_white=False,
                candidate=candidate_engine,
                baseline=baseline_engine,
                movetime_ms=args.movetime,
                timeout_s=args.timeout,
            )
            stats.add_pair(first, second)

            print(
                f"[pair {pair_index + 1:>3}/{total_pairs}] "
                f"G1={first.winner:<9} ({first.reason}), "
                f"G2={second.winner:<9} ({second.reason})"
            )
            if "legal move" in first.reason:
                print_failure_debug("G1", first)
            if "legal move" in second.reason:
                print_failure_debug("G2", second)

            if (pair_index + 1) % max(1, args.report_every) == 0 or (pair_index + 1) == total_pairs:
                print()
                print(format_summary(stats, args.confidence, args.candidate_name, baseline_name))
                verdict = verdict_from_stats(stats, args.confidence, args.min_pairs)
                print(f"Verdict: {verdict}")
                print()
                if args.early_stop and verdict in {"PASS", "FAIL"}:
                    break

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as exc:
        print(f"\nValidation failed: {exc}", file=sys.stderr)
        return 1
    finally:
        candidate_engine.close()
        baseline_engine.close()

    final_verdict = verdict_from_stats(stats, args.confidence, args.min_pairs)
    print("Final summary")
    print("-------------")
    print(format_summary(stats, args.confidence, args.candidate_name, baseline_name))
    print(f"Verdict: {final_verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
