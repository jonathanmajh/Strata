"""A/B two engine builds or environments on one config: restart the server per arm, alternate arms over rounds.

    python tools/ab_engine.py strata-q2_0.json A:exe=/path/a B:exe=/path/b,STRATA_EXP_MODE=0 [--rounds 3]

An arm is NAME:key=value,...; `exe` replaces the config's engine, `xargs` appends ';'-separated engine arguments, every other key is set in the engine's environment.  Any number of arms may be given; the arms rotate through positions (Latin shift), so every arm leads equally often and takes every position.  Each round starts the server once per arm, warms it up with one short answer (wait_ready posts "Say hi."), then sends the same requests: greedy, fixed prompts, a short chat set and one or more long prompts (`--long-words-list 6000,24000`).  Prints the server's own timings (llama.cpp's `timings`: decode and prompt tokens per second) per request, per-arm medians with min..max spread, and samples VRAM (rocm-smi) after each arm's ready and at the end of its requests.  Compare medians over rounds.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = (
    "Write a Python function that merges two sorted lists into one sorted list, with a docstring and two tests.",
    "Explain in two paragraphs how a refrigerator moves heat from inside to outside.",
    "List ten practical tips for learning a new language as an adult, one sentence each.",
)


def long_prompt(words: int) -> str:
    rng = random.Random(7)
    vocab = ("river stone market lantern copper winter garden signal harbor needle orbit velvet canyon ember "
             "ledger quartz meadow anchor violet thunder").split()
    body = " ".join(rng.choice(vocab) for _ in range(words))
    return f"Here is a list of words:\n{body}\n\nSummarize what kinds of words appear in the list, in five sentences."


def post(port: int, prompt: str, max_tokens: int) -> dict:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps({"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
                         "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.loads(r.read())


def vram() -> dict[str, int]:
    try:
        out = subprocess.check_output(["rocm-smi", "--showmeminfo", "vram"], text=True, timeout=60)
    except Exception as e:
        print(f"  vram sample failed: {e}", flush=True)
        return {}
    m: dict[str, int] = {}
    for match in re.finditer(r"GPU\[(\d+)\]\s+: VRAM Total Used Memory \(B\): (\d+)", out):
        m[f"vram_used_gpu{match.group(1)}"] = int(match.group(2))
    return m


def wait_ready(port: int, proc: subprocess.Popen, timeout: float = 1800) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with {proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as r:
                if r.status == 200:
                    post(port, "Say hi.", 8)
                    return
        except Exception:
            pass
        time.sleep(3)
    raise TimeoutError("server not ready")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("arms", nargs="+")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--long-words", type=int, default=6000)
    ap.add_argument("--long-words-list", default="")
    ap.add_argument("--long-max-new", type=int, default=128)
    ap.add_argument("--log", default="ab_engine.jsonl")
    a = ap.parse_args()
    base = json.loads(Path(a.config).read_text())
    arms = []
    for s in a.arms:
        name, _, kv = s.partition(":")
        env = dict(x.split("=", 1) for x in kv.split(",") if x)
        arms.append((name, env))
    longs = [int(x) for x in a.long_words_list.split(",") if x.strip()] or [a.long_words]
    reqs = [(f"chat{i}", p, a.max_new) for i, p in enumerate(PROMPTS)]
    if len(longs) == 1:
        reqs.append(("long", long_prompt(longs[0]), a.long_max_new))
    else:
        for i, w in enumerate(longs):
            reqs.append((f"long{i}", long_prompt(w), a.long_max_new))
    out = open(a.log, "a")
    res: dict = {n: {} for n, _ in arms}
    for rnd in range(a.rounds):
        order = arms[rnd % len(arms):] + arms[: rnd % len(arms)]
        print("order round %d: %s" % (rnd, " ".join(n for n, _ in order)), flush=True)
        for name, env in order:
            cfg = dict(base)
            e = dict(env)
            if "exe" in e:
                cfg["exe"] = e.pop("exe")
            if "xargs" in e:                       # extra engine arguments, ';'-separated
                cfg["args"] = list(base["args"]) + e.pop("xargs").split(";")
            cfg["port"] = a.port
            cfg["log"] = str(Path(tempfile.gettempdir()) / f"ab_{name}_r{rnd}.log")
            cpath = Path(tempfile.gettempdir()) / f"ab_{name}.json"
            cpath.write_text(json.dumps(cfg, indent=1))
            penv = dict(os.environ, **e)
            proc = subprocess.Popen([sys.executable, str(ROOT / "serve" / "server.py"), "--engine", "strata", "--config",
                                     str(cpath), "--port", str(a.port)], cwd=str(ROOT), env=penv,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                t0 = time.time()
                wait_ready(a.port, proc)
                ready_vram = vram()
                print(f"round {rnd} {name}: ready in {time.time() - t0:.0f} s" +
                      (("  vram " + ",".join(f"g{k[-1]}={v // (1 << 20)} MB" for k, v in sorted(ready_vram.items()))) if ready_vram else ""),
                      flush=True)
                for tag, p, mx in reqs:
                    r = post(a.port, p, mx)
                    t = r.get("timings", {})
                    row = {"round": rnd, "arm": name, "req": tag, "decode": t.get("predicted_per_second"),
                           "prompt": t.get("prompt_per_second"), "prompt_n": t.get("prompt_n"),
                           "n": r.get("usage", {}).get("completion_tokens"),
                           **ready_vram,
                           "text": r["choices"][0]["message"].get("content", "")[:80]}
                    out.write(json.dumps(row) + "\n")
                    out.flush()
                    res[name].setdefault(tag, []).append(row)
                    print(f"  {tag:6s} decode {row['decode']} tok/s  prompt {row['prompt']} tok/s ({row['prompt_n']})  "
                          f"n {row['n']}", flush=True)
                end_vram = vram()
                if end_vram:
                    print(f"  vram end " + ",".join(f"g{k[-1]}={v // (1 << 20)} MB" for k, v in sorted(end_vram.items())),
                          flush=True)
            finally:
                proc.terminate()
                try:
                    proc.wait(60)
                except subprocess.TimeoutExpired:
                    proc.kill()
                time.sleep(5)
    print("\nmedians (n=%d) with spread | outlier flag at >2x from arm's median:" % a.rounds)
    for tag, _, _ in reqs:
        cells = []
        for name, _ in arms:
            rows = res[name].get(tag, [])
            ds = [x["decode"] for x in rows if x["decode"]]
            ps = [x["prompt"] for x in rows if x["prompt"]]
            if not rows:
                cells.append(f"{name}: no data")
                continue
            md = statistics.median(ds) if ds else float("nan")
            mp = statistics.median(ps) if ps else float("nan")
            sd = f"{min(ds):.1f}..{max(ds):.1f}" if ds else "n/a"
            sp = f"{min(ps):.1f}..{max(ps):.1f}" if ps else "n/a"
            cells.append(f"{name}: decode {md:6.1f} [{sd}]  prompt {mp:7.1f} [{sp}]")
        print(f"  {tag:6s} | " + " | ".join(cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
