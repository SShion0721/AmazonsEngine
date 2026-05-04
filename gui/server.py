from __future__ import annotations

import os
import queue
import subprocess
import threading
import time
from typing import Callable

from flask import Flask, jsonify, request, send_file

app = Flask(__name__)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
GUI_ROOT = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(GUI_ROOT, 'index.html')
BUILD_ENGINE_PATH = os.path.join(REPO_ROOT, 'build', 'amazons.exe')
ROOT_ENGINE_PATH = os.path.join(REPO_ROOT, 'amazons.exe')

engine_proc: subprocess.Popen[str] | None = None
engine_lock = threading.Lock()
engine_output: queue.Queue[str] = queue.Queue()


def resolve_engine_path() -> str:
    if os.path.exists(BUILD_ENGINE_PATH):
        return BUILD_ENGINE_PATH
    return ROOT_ENGINE_PATH


def drain_output() -> None:
    while True:
        try:
            engine_output.get_nowait()
        except queue.Empty:
            break


def _reader_loop(proc: subprocess.Popen[str], output_queue: queue.Queue[str]) -> None:
    assert proc.stdout is not None
    for raw_line in proc.stdout:
        output_queue.put(raw_line.rstrip('\r\n'))


def _ensure_engine_running() -> None:
    global engine_proc, engine_output

    if engine_proc is not None and engine_proc.poll() is None:
        return

    engine_path = resolve_engine_path()
    if not os.path.exists(engine_path):
        raise RuntimeError(f'Engine executable not found: {engine_path}')

    engine_proc = subprocess.Popen(
        [engine_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=REPO_ROOT,
        text=True,
        encoding='utf-8',
        errors='replace',
        bufsize=1,
    )

    engine_output = queue.Queue()
    threading.Thread(target=_reader_loop, args=(engine_proc, engine_output), daemon=True).start()

    send_command('uci')
    uci_lines = _read_until(lambda line: line == 'uciok', timeout=5.0)
    if not any(line == 'uciok' for line in uci_lines):
        raise RuntimeError('Engine did not finish UCI handshake')

    send_command('isready')
    ready_lines = _read_until(lambda line: line == 'readyok', timeout=5.0)
    if not any(line == 'readyok' for line in ready_lines):
        raise RuntimeError('Engine did not become ready')


def send_command(command: str) -> None:
    _ensure_engine_running()
    assert engine_proc is not None and engine_proc.stdin is not None
    engine_proc.stdin.write(command + '\n')
    engine_proc.stdin.flush()


def _read_until(condition: Callable[[str], bool], timeout: float) -> list[str]:
    deadline = time.time() + timeout
    lines: list[str] = []

    while time.time() < deadline:
        remaining = deadline - time.time()
        try:
            line = engine_output.get(timeout=max(0.01, remaining))
        except queue.Empty:
            break

        if not line:
            continue

        lines.append(line)
        if condition(line):
            break

    return lines


def _parse_uci_lines(lines: list[str]) -> dict:
    payload = {'name': 'Unknown Engine', 'author': 'Unknown', 'options': []}

    for line in lines:
        if line.startswith('id name '):
            payload['name'] = line[len('id name '):]
            continue
        if line.startswith('id author '):
            payload['author'] = line[len('id author '):]
            continue
        if not line.startswith('option name '):
            continue

        tokens = line.split()
        option = {'name': '', 'type': '', 'default': '', 'current': '', 'min': None, 'max': None}
        idx = 2
        while idx < len(tokens) and tokens[idx] != 'type':
            option['name'] += ('' if not option['name'] else ' ') + tokens[idx]
            idx += 1

        if idx + 1 >= len(tokens):
            continue
        idx += 1
        option['type'] = tokens[idx]
        idx += 1

        while idx < len(tokens):
            key = tokens[idx]
            idx += 1
            if idx >= len(tokens):
                break
            if key == 'default':
                option['default'] = tokens[idx]
                option['current'] = tokens[idx]
            elif key == 'min':
                option['min'] = int(tokens[idx])
            elif key == 'max':
                option['max'] = int(tokens[idx])
            idx += 1

        payload['options'].append(option)

    return payload


def get_engine_runtime_state() -> dict:
    with engine_lock:
        _ensure_engine_running()
        drain_output()
        send_command('eval')
        lines = _read_until(lambda line: line.startswith('Total     :'), timeout=5.0)

    state = {
        'mode': 'Unknown',
        'classical': None,
        'nnue': None,
        'total': None,
        'nnue_active': False,
        'nnue_reason': '',
    }

    for line in lines:
        if line.startswith('Classical :'):
            try:
                state['classical'] = int(line.split(':', 1)[1].strip())
            except ValueError:
                pass
        elif line.startswith('NNUE      :'):
            payload = line.split(':', 1)[1].strip()
            if ' (' in payload:
                value_text, suffix = payload.split(' (', 1)
                state['nnue_reason'] = suffix.rstrip(')')
            else:
                value_text = payload
            try:
                state['nnue'] = int(value_text.strip())
            except ValueError:
                pass
        elif line.startswith('Mode      :'):
            state['mode'] = line.split(':', 1)[1].strip()
        elif line.startswith('Total     :'):
            rest = line.split(':', 1)[1].strip()
            score_text = rest.split(' ', 1)[0]
            try:
                state['total'] = int(score_text)
            except ValueError:
                pass

    state['nnue_active'] = 'NNUE' in state['mode']
    return state


def get_engine_description() -> dict:
    with engine_lock:
        _ensure_engine_running()
        drain_output()
        send_command('uci')
        lines = _read_until(lambda line: line == 'uciok', timeout=5.0)
        payload = _parse_uci_lines(lines)
    payload['state'] = get_engine_runtime_state()
    return payload


def apply_engine_option(name: str, value: str | None) -> list[str]:
    with engine_lock:
        _ensure_engine_running()
        drain_output()
        if value is None:
            send_command(f'setoption name {name}')
        else:
            send_command(f'setoption name {name} value {value}')
        send_command('isready')
        return _read_until(lambda line: line == 'readyok', timeout=10.0)


def get_bestmove(position_cmd: str, go_cmd: str, timeout: float) -> tuple[str | None, list[str]]:
    with engine_lock:
        _ensure_engine_running()
        drain_output()
        send_command(position_cmd)
        send_command(go_cmd)
        lines = _read_until(lambda line: line.startswith('bestmove'), timeout=timeout)

    bestmove = None
    info_lines: list[str] = []
    for line in lines:
        if line.startswith('bestmove'):
            parts = line.split()
            bestmove = parts[1] if len(parts) > 1 else None
        elif line.startswith('info'):
            info_lines.append(line)

    return bestmove, info_lines


@app.route('/')
def index():
    return send_file(INDEX_PATH)


@app.route('/api/newgame', methods=['POST'])
def new_game():
    with engine_lock:
        _ensure_engine_running()
        drain_output()
        send_command('ucinewgame')
        send_command('isready')
        lines = _read_until(lambda line: line == 'readyok', timeout=5.0)
    return jsonify({'status': 'ok', 'ready': any(line == 'readyok' for line in lines)})


@app.route('/api/engine/options', methods=['GET'])
def engine_options():
    try:
        return jsonify(get_engine_description())
    except Exception as exc:  # pragma: no cover - runtime protection for GUI
        return jsonify({'error': str(exc), 'options': []}), 500


@app.route('/api/engine/state', methods=['GET'])
def engine_state():
    try:
        return jsonify(get_engine_runtime_state())
    except Exception as exc:  # pragma: no cover - runtime protection for GUI
        return jsonify({'error': str(exc)}), 500


@app.route('/api/engine/option', methods=['POST'])
def set_engine_option():
    data = request.get_json(silent=True) or {}
    name = str(data.get('name', '')).strip()
    option_type = str(data.get('type', '')).strip()
    value = data.get('value')

    if not name:
        return jsonify({'error': 'Missing option name'}), 400

    try:
        lines = apply_engine_option(name, None if option_type == 'button' else str(value))
    except Exception as exc:  # pragma: no cover - runtime protection for GUI
        return jsonify({'error': str(exc)}), 500

    info_lines = [line for line in lines if line.startswith('info string')]
    return jsonify({'status': 'ok', 'messages': info_lines})


@app.route('/api/move', methods=['POST'])
def make_move():
    data = request.get_json(silent=True) or {}
    moves = data.get('moves', [])
    think_time = int(data.get('thinkTime', 2000))

    if moves:
        position_cmd = 'position startpos moves ' + ' '.join(moves)
    else:
        position_cmd = 'position startpos'

    bestmove, info_lines = get_bestmove(position_cmd, f'go movetime {think_time}',
                                        timeout=max(10.0, think_time / 1000.0 + 5.0))
    if bestmove is None:
        return jsonify({'error': 'Engine did not return a bestmove within the timeout'}), 504

    depth = 0
    score = 0
    nodes = 0
    for info in info_lines:
        parts = info.split()
        for i, token in enumerate(parts):
            if token == 'depth' and i + 1 < len(parts):
                depth = int(parts[i + 1])
            elif token == 'cp' and i + 1 < len(parts):
                score = int(parts[i + 1])
            elif token == 'nodes' and i + 1 < len(parts):
                nodes = int(parts[i + 1])

    return jsonify({
        'bestmove': bestmove,
        'depth': depth,
        'score': score,
        'nodes': nodes,
    })


if __name__ == '__main__':
    print('正在启动亚马逊棋引擎...')
    _ensure_engine_running()
    print(f'引擎已就绪：{resolve_engine_path()}')
    print('请在浏览器中打开: http://localhost:5000')
    app.run(host='0.0.0.0', port=5000, debug=False)
