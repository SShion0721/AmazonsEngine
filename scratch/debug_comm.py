import subprocess
import threading
import queue
import time
import os

REPO_ROOT = r'e:\Desktop\AmazonsEngine'
ENGINE_PATH = os.path.join(REPO_ROOT, 'build', 'amazons.exe')

def _reader_loop(proc, output_queue):
    for raw_line in proc.stdout:
        print(f"DEBUG: ENGINE OUT: {raw_line.strip()}")
        output_queue.put(raw_line.rstrip('\r\n'))

def test_engine():
    print(f"Testing engine at: {ENGINE_PATH}")
    proc = subprocess.Popen(
        [ENGINE_PATH],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=REPO_ROOT,
        text=True,
        encoding='utf-8',
        errors='replace',
        bufsize=1,
    )
    
    q = queue.Queue()
    t = threading.Thread(target=_reader_loop, args=(proc, q), daemon=True)
    t.start()
    
    time.sleep(1)
    print("Sending 'uci'...")
    proc.stdin.write('uci\n')
    proc.stdin.flush()
    
    deadline = time.time() + 5.0
    found_uciok = False
    while time.time() < deadline:
        try:
            line = q.get(timeout=0.1)
            if line == 'uciok':
                found_uciok = True
                break
        except queue.Empty:
            if proc.poll() is not None:
                print(f"Engine exited with code {proc.returncode}")
                break
            continue
            
    if found_uciok:
        print("UCI Handshake SUCCESS")
    else:
        print("UCI Handshake FAILED")
        
    proc.terminate()

if __name__ == "__main__":
    test_engine()
