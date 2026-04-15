"""
7-Hour Training Orchestrator v2 - Ultra-conservative for RTX 5070
-----------------------------------------------------------------
v2 changes (post cudaErrorUnknown crash at step 213):
  - GPU power: 130W (52%) down from 165W
  - Cooldown: 90s between chunks (up from 60s)
  - 25 steps/chunk (down from 50)
  - Additional VRAM cleanup between chunks
  - Pre-chunk CUDA health check

Strategy:
  - ~2 min training chunk -> 90s GPU cooldown (CUDA context fully destroyed)
  - If GPU > 75C after chunk -> extend cooldown until < 65C
  - On crash: exponential backoff (30s -> 60s -> 120s -> 240s), then retry
  - Max 5 consecutive crashes before circuit-breaker (5min pause)
  - GPU power capped at 130W (52%) for maximum stability
  - Total time limit: 7 hours hard stop

Exit codes from train_7h.py:
  0 = Training complete
  1 = Error/crash
  2 = More chunks needed
  3 = GPU too hot (retry after cooldown)
"""
import subprocess
import sys
import time
import os
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = str(ROOT / "venv" / "Scripts" / "python.exe")
TRAINER = str(ROOT / "train_7h.py")
LOG_FILE = ROOT / "training_log.txt"

# ─── CONFIG ───────────────────────────────────────────────
TOTAL_HOURS       = 7
COOLDOWN_SECS     = 90      # v2: up from 60 — more GPU rest
HOT_COOLDOWN_SECS = 30      # Extra wait per degree above target
TARGET_COOL_TEMP  = 65      # v2: down from 70 — wait until cooler
MAX_GPU_TEMP      = 75      # v2: down from 78 — stricter threshold
GPU_POWER_LIMIT   = 130     # v2: down from 165 (52% of 250W)
GPU_POWER_RESET   = 250     # Watts (restore on finish)
MAX_CONSECUTIVE_CRASHES = 5
CIRCUIT_BREAKER_SECS = 300  # 5 min pause after too many crashes
CRASH_BACKOFF_BASE = 30     # Exponential backoff base


def log(msg):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] [ORCH] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except:
        pass


def get_gpu_temp():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        return int(r.stdout.strip())
    except:
        return None


def set_gpu_power(watts):
    try:
        subprocess.run(
            ["nvidia-smi", "-pl", str(watts)],
            capture_output=True, timeout=5,
        )
        log(f"GPU power limit set to {watts}W")
    except Exception as e:
        log(f"Failed to set GPU power: {e}")


def wait_for_cool_gpu(target_temp=TARGET_COOL_TEMP, max_wait=300):
    """Wait until GPU temperature drops below target. Max wait in seconds."""
    temp = get_gpu_temp()
    if temp is None or temp <= target_temp:
        return

    log(f"GPU at {temp}°C — cooling down (target: <{target_temp}°C)...")
    waited = 0
    while waited < max_wait:
        time.sleep(10)
        waited += 10
        temp = get_gpu_temp()
        if temp is None:
            break
        if temp <= target_temp:
            log(f"GPU cooled to {temp}°C after {waited}s")
            return
        if waited % 30 == 0:
            log(f"  Still cooling... {temp}°C ({waited}s elapsed)")

    final_temp = get_gpu_temp()
    log(f"Cooldown timeout after {max_wait}s — GPU at {final_temp}°C, proceeding anyway")


def run_chunk():
    """Run one training chunk as a subprocess. Returns exit code."""
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # v2: deterministic GPU ordering

    try:
        result = subprocess.run(
            [PYTHON, TRAINER],
            cwd=str(ROOT),
            env=env,
            timeout=900,  # v2: 15 min max per chunk (was 30min, chunks are smaller now)
        )
        return result.returncode
    except subprocess.TimeoutExpired:
        log("[TIMEOUT] Chunk exceeded 15min — killing")
        return 1
    except Exception as e:
        log(f"[ERROR] Failed to launch chunk: {e}")
        return 1


def get_current_step():
    """Read current step from checkpoints directory."""
    ckpt_dir = ROOT / "data" / "models" / "social_good_v1"
    if not ckpt_dir.exists():
        return 0
    steps = []
    for d in ckpt_dir.iterdir():
        if d.is_dir() and d.name.startswith("checkpoint-"):
            try:
                steps.append(int(d.name.split("-")[1]))
            except ValueError:
                pass
    return max(steps) if steps else 0


def main():
    start_time = datetime.now()
    deadline = start_time + timedelta(hours=TOTAL_HOURS)

    log("=" * 60)
    log(f"  7-HOUR TRAINING SESSION STARTED")
    log(f"  Start:    {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  Deadline: {deadline.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  GPU Power Limit: {GPU_POWER_LIMIT}W")
    log(f"  Cooldown: {COOLDOWN_SECS}s between chunks")
    log(f"  Current step: {get_current_step()}/1701")
    log("=" * 60)

    # Set conservative GPU power
    set_gpu_power(GPU_POWER_LIMIT)

    consecutive_crashes = 0
    total_chunks = 0
    total_crashes = 0

    while datetime.now() < deadline:
        remaining_time = deadline - datetime.now()
        remaining_min = remaining_time.total_seconds() / 60
        current_step = get_current_step()

        if remaining_min < 5:
            log(f"Less than 5 minutes remaining — stopping gracefully")
            break

        log(f"─── Chunk #{total_chunks + 1} | Step {current_step}/1701 | "
            f"{remaining_min:.0f}min remaining ───")

        # Pre-flight: temperature check
        temp = get_gpu_temp()
        if temp is not None and temp > MAX_GPU_TEMP:
            log(f"GPU too hot ({temp}°C) — waiting for cooldown...")
            wait_for_cool_gpu()

        # Run the chunk
        exit_code = run_chunk()
        total_chunks += 1

        if exit_code == 0:
            # Training complete!
            log("=" * 60)
            log("  TRAINING COMPLETE! All steps finished.")
            log(f"  Total chunks: {total_chunks}")
            log(f"  Total crashes recovered: {total_crashes}")
            log(f"  Total time: {(datetime.now() - start_time).total_seconds()/3600:.1f}h")
            log("=" * 60)
            set_gpu_power(GPU_POWER_RESET)
            return 0

        elif exit_code == 2:
            # More chunks needed — normal cooldown
            consecutive_crashes = 0
            log(f"  Chunk OK — {COOLDOWN_SECS}s cooldown...")
            time.sleep(COOLDOWN_SECS)

            # Post-cooldown temperature check
            wait_for_cool_gpu()

        elif exit_code == 3:
            # GPU too hot — extended cooldown
            log(f"  GPU overheated — extended cooldown...")
            wait_for_cool_gpu(target_temp=60, max_wait=300)  # v2: wait until 60°C

        elif exit_code == 1:
            # Crash!
            consecutive_crashes += 1
            total_crashes += 1
            log(f"  [CRASH #{total_crashes}] Consecutive: {consecutive_crashes}/{MAX_CONSECUTIVE_CRASHES}")

            if consecutive_crashes >= MAX_CONSECUTIVE_CRASHES:
                log(f"  [CIRCUIT BREAKER] {MAX_CONSECUTIVE_CRASHES} consecutive crashes!")
                log(f"  Pausing {CIRCUIT_BREAKER_SECS}s then resetting counter...")
                time.sleep(CIRCUIT_BREAKER_SECS)
                wait_for_cool_gpu(target_temp=60, max_wait=300)
                consecutive_crashes = 0
            else:
                # Exponential backoff
                backoff = CRASH_BACKOFF_BASE * (2 ** (consecutive_crashes - 1))
                backoff = min(backoff, 300)  # Cap at 5 min
                log(f"  Backoff: {backoff}s before retry...")
                time.sleep(backoff)
                wait_for_cool_gpu()

        else:
            # Unknown exit code
            log(f"  Unknown exit code: {exit_code} — treating as crash")
            consecutive_crashes += 1
            total_crashes += 1
            time.sleep(60)

    # Time's up
    final_step = get_current_step()
    elapsed_h = (datetime.now() - start_time).total_seconds() / 3600

    log("=" * 60)
    log(f"  7-HOUR SESSION ENDED")
    log(f"  Final step: {final_step}/1701 ({100*final_step/1701:.1f}%)")
    log(f"  Total chunks: {total_chunks}")
    log(f"  Total crashes recovered: {total_crashes}")
    log(f"  Actual time: {elapsed_h:.1f}h")
    log("=" * 60)

    set_gpu_power(GPU_POWER_RESET)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log("\n[ABORT] Ctrl+C — stopping gracefully")
        set_gpu_power(GPU_POWER_RESET)
        sys.exit(130)
    except Exception as e:
        log(f"[FATAL] Orchestrator crashed: {e}")
        set_gpu_power(GPU_POWER_RESET)
        sys.exit(1)
