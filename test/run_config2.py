"""Config 2: Bounded(6,4) + Null Move + No Categories"""
import subprocess, sys, os, time, json, math, threading
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(os.path.join(project_root, "Amazon", "src", "ai", "src", "build"))
sys.path.append(os.path.join(project_root, "Amazon", "src", "ai", "src2", "build"))

try:
    import amazon_ai
except ImportError:
    import amazon_ai_test as amazon_ai

E, WQ, BQ, ST = 0, 1, 2, 3
W, B = 1, -1
INIT_W = [(3,0),(0,3),(0,6),(3,9)]
INIT_B = [(6,0),(9,3),(9,6),(6,9)]
GAME_TIMEOUT = 120
ENGINE_PATH = os.path.join(project_root, "amazons.exe")

def init_board():
    g = np.zeros((10,10), dtype=np.int32)
    for r,c in INIT_W: g[r][c] = WQ
    for r,c in INIT_B: g[r][c] = BQ
    return g, [[30,3,6,39], [60,93,96,69]]

def sq2str(r, c): return f"{chr(97+c)}{r+1}"
def str2sq(s): return (int(s[1:])-1, ord(s[0].lower())-97)
def parse_uci(s):
    fr, rest = s.split("-")
    to, ar = rest.split("/")
    return str2sq(fr), str2sq(to), str2sq(ar)

def path_clear(g, r1, c1, r2, c2):
    dr, dc = r2-r1, c2-c1
    if dr == 0 and dc == 0: return False
    if not (dr==0 or dc==0 or abs(dr)==abs(dc)): return False
    n = max(abs(dr), abs(dc))
    sr = 0 if dr==0 else (1 if dr>0 else -1)
    sc = 0 if dc==0 else (1 if dc>0 else -1)
    for i in range(1, n):
        if g[r1+sr*i][c1+sc*i] != E: return False
    return True

def valid_move(g, wq, bq, side, fr, to, ar):
    r1,c1 = fr; r2,c2 = to; ra,ca = ar
    queens = wq if side==W else bq
    if r1*10+c1 not in queens: return False
    if not path_clear(g, r1, c1, r2, c2): return False
    if g[r2][c2] != E: return False
    saved = g[r1][c1]; g[r1][c1] = E
    ok = path_clear(g, r2, c2, ra, ca)
    if ok and (ra,ca)!=(r1,c1) and g[ra][ca] != E: ok = False
    g[r1][c1] = saved
    return ok

def apply_move(g, wq, bq, side, fr, to, ar):
    qt = WQ if side==W else BQ
    g[fr[0]][fr[1]] = E
    g[to[0]][to[1]] = qt
    g[ar[0]][ar[1]] = ST
    q = wq if side==W else bq
    q[q.index(fr[0]*10+fr[1])] = to[0]*10+to[1]

def has_move(g, wq, bq, side):
    queens = wq if side==W else bq
    for qp in queens:
        qr, qc = qp//10, qp%10
        for dr in (-1,0,1):
            for dc in (-1,0,1):
                if dr==0 and dc==0: continue
                for d in range(1,10):
                    nr, nc = qr+dr*d, qc+dc*d
                    if not(0<=nr<10 and 0<=nc<10): break
                    if g[nr][nc]!=E: break
                    for ar in (-1,0,1):
                        for ac in (-1,0,1):
                            if ar==0 and ac==0: continue
                            for ad in range(1,10):
                                er, ec = nr+ar*ad, nc+ac*ad
                                if not(0<=er<10 and 0<=ec<10): break
                                if g[er][ec]!=E:
                                    if er==qr and ec==qc: return True
                                    break
                                return True
    return False

def uci_go(proc, uci_moves, ms):
    mv = " ".join(uci_moves)
    pos = f"position startpos moves {mv}" if mv else "position startpos"
    proc.stdin.write((pos + "\n").encode()); proc.stdin.flush()
    proc.stdin.write((f"go movetime {ms}\n").encode()); proc.stdin.flush()
    result = [None]
    def reader():
        try:
            while True:
                raw = proc.stdout.readline()
                if not raw: break
                l = raw.decode(errors="replace").strip()
                if l.startswith("bestmove"):
                    parts = l.split()
                    if len(parts) >= 2 and parts[1] != "none":
                        result[0] = parts[1]
                    break
        except: pass
    t = threading.Thread(target=reader, daemon=True)
    t.start()
    t.join(timeout=ms/1000 + 5)
    return result[0]

def play_game(gn, ms, uci_white):
    g, [wq, bq] = init_board()
    uci_side = W if uci_white else B
    uci_moves = []
    n = 0
    proc = subprocess.Popen(
        [ENGINE_PATH], stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0
    )
    def drain():
        try:
            while proc.stderr.read(4096): pass
        except: pass
    threading.Thread(target=drain, daemon=True).start()
    proc.stdin.write(b"uci\n"); proc.stdin.flush()
    s = time.time()
    while time.time()-s < 10:
        l = proc.stdout.readline()
        if not l: return (gn, -1, -2)
        if "uciok" in l.decode(errors="replace"): break
    else:
        return (gn, -1, -2)
    for k, v in [("Search Mode","Match"),("Use MCTS Root","false"),
                  ("Use Strong Classical","true"),("Eval Tier","Fast"),
                  ("Threads","1"),("Hash","512"),
                  ("Use Bounded MoveGen","true"),("Bounded Dest Cap","6"),
                  ("Bounded Arrow Cap","4"),
                  ("Use Null Move","true"),("Use Move Categories","false")]:
        proc.stdin.write(f"setoption name {k} value {v}\n".encode()); proc.stdin.flush()
    proc.stdin.write(b"ucinewgame\n"); proc.stdin.flush()
    proc.stdin.write(b"isready\n"); proc.stdin.flush()
    while True:
        l = proc.stdout.readline()
        if not l: return (gn, -1, -2)
        if "readyok" in l.decode(errors="replace"): break
    ai = amazon_ai.AmazonasAI()
    while n < 200:
        side = W if n % 2 == 0 else B
        if not has_move(g, wq, bq, side):
            try: proc.stdin.write(b"quit\n"); proc.stdin.flush()
            except: pass
            proc.wait(timeout=3)
            return (gn, 1 if -side == uci_side else -1, n)
        ok = False
        for _ in range(8):
            try:
                if side == uci_side:
                    s = uci_go(proc, uci_moves, ms)
                    if s is None: continue
                    fr, to, ar = parse_uci(s)
                else:
                    arr = np.array(g, dtype=np.int32)
                    qp = [list(wq), list(bq)]
                    r = ai.uct_search(arr, qp, -1 if side==B else 1, ms/1000.0, False)
                    fr = (r.From//10, r.From%10)
                    to = (r.To//10, r.To%10)
                    ar = (r.Stone//10, r.Stone%10)
                    s = f"{sq2str(*fr)}-{sq2str(*to)}/{sq2str(*ar)}"
            except:
                ai = amazon_ai.AmazonasAI()
                continue
            if valid_move(g, wq, bq, side, fr, to, ar):
                apply_move(g, wq, bq, side, fr, to, ar)
                uci_moves.append(s)
                ok = True
                break
            else:
                ai = amazon_ai.AmazonasAI()
        if not ok:
            try: proc.stdin.write(b"quit\n"); proc.stdin.flush()
            except: pass
            try: proc.kill()
            except: pass
            return (gn, 1 if -side == uci_side else -1, n)
        n += 1
    try: proc.stdin.write(b"quit\n"); proc.stdin.flush()
    except: pass
    try: proc.kill()
    except: pass
    return (gn, 0, n)

def calc_elo(results):
    valid = [(r,m) for r,m in results if m >= 0]
    if not valid: return 0, 0
    w = sum(1 for r,_ in valid if r==1)
    l = sum(1 for r,_ in valid if r==-1)
    d = sum(1 for r,_ in valid if r==0)
    n = len(valid)
    s = (w+d*0.5)/n
    if s<=0: return -400,0
    if s>=1: return 400,0
    elo = -400*math.log10(1/s-1)
    se = 400/math.log(10)*math.sqrt(s*(1-s)/n)/(s*(1-s))
    return elo, se

def main():
    MS = 100
    GAMES = 100
    PAR = 12
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = os.path.join(current_dir, f"config2_log_{ts}.txt")
    resfile = os.path.join(current_dir, f"config2_results_{ts}.json")
    print("Config 2: Bounded(6,4) + Null Move + No Categories")
    t0 = time.time()
    results = [None]*GAMES
    with open(logfile,"w") as f:
        f.write(f"Config 2 | {MS}ms/move | {GAMES} games\n")
    def run_game(gn):
        uci_white = (gn % 2 == 0)
        return play_game(gn, MS, uci_white)
    with ThreadPoolExecutor(max_workers=PAR) as pool:
        futures = {pool.submit(run_game, gn): gn for gn in range(GAMES)}
        done_count = 0
        for fut in as_completed(futures):
            gn, res, nm = fut.result()
            results[gn] = (res, nm)
            done_count += 1
            if nm == -2: tag = "ERR!"
            elif nm < 0: tag = "TMO!"
            else: tag = "E1+" if res==1 else "E2+" if res==-1 else "Draw"
            done = [r for r in results if r is not None]
            w1 = sum(1 for r,_ in done if r==1 and _>=0)
            w2 = sum(1 for r,_ in done if r==-1 and _>=0)
            print(f"[{done_count:3d}/{GAMES}] G{gn+1:3d} {tag} E1 {w1}-{w2} E2 | {time.time()-t0:.0f}s")
            with open(logfile,"a") as f:
                f.write(f"G{gn+1}: {tag} E1={gn%2==0}\n")
    elapsed = time.time()-t0
    valid = [(r,m) for r,m in results if m >= 0]
    gr = [r for r,_ in valid]
    w1 = sum(1 for r in gr if r==1)
    w2 = sum(1 for r in gr if r==-1)
    dr = sum(1 for r in gr if r==0)
    elo, se = calc_elo(results)
    sc = (w1+dr*0.5)/len(gr)*100 if gr else 0
    print(f"\nConfig 2: {w1}W {dr}D {w2}L | Elo {elo:+.1f} +/- {se:.1f} | Score {sc:.1f}%")
    with open(resfile,"w") as f:
        json.dump({"config":"Bounded(6,4)+NullMove+NoCategories","w1":w1,"w2":w2,"draws":dr,"elo":elo,"se":se,"score":sc}, f, indent=2)

if __name__ == "__main__":
    main()
