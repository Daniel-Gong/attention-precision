"""Run LLM studies (L5 intervene, L7 locate, L2/L3 if needed) on Modal GPUs, one container per model.

Run from the repo root on a machine with a Modal token (Misha login node):
  modal run scripts/modal_run.py::main --study intervene         # every model with L3 results
  modal run scripts/modal_run.py::main --study locate --models "gpt2 EleutherAI/pythia-410m"

Inputs are read locally (results/parts/l3_*.jsonl, l2_*.jsonl) and passed to the container;
each container's output JSONL is written back to results/parts/<prefix>_<model>.jsonl and its
log to logs/modal_<study>_<model>.log. Hugging Face downloads persist in the Modal volume
"ap-hf-cache". Models of 6B parameters or more get an H100, the rest an L40S.
"""
import glob
import os
import pathlib

import modal
import modal.exception

ROOT = pathlib.Path(__file__).resolve().parent.parent
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.11.0", "transformers==5.19.0", "numpy", "scipy", "pyyaml", "accelerate")
         .add_local_dir(str(ROOT / "configs"), "/root/configs")
         .add_local_dir(str(ROOT / "src"), "/root/src"))
app = modal.App("attention-precision", image=image)
hf = modal.Volume.from_name("ap-hf-cache", create_if_missing=True)

STUDY = {  # study -> (module, output prefix, extra args)
    "intervene": ("ap.llm.intervene", "l5", ["--l3", "/tmp/l3.jsonl"]),
    "locate": ("ap.llm.locate", "l7", ["--l3", "/tmp/l3.jsonl", "--l2", "/tmp/l2.jsonl", "--ns", "2", "3", "4"]),
    "heads": ("ap.llm.heads", "l3", ["--ns", "1", "2", "3", "4"]),
    "behave": ("ap.llm.behave", "l2", []),
    "suppress": ("ap.llm.suppress", "l5s", []),
    "probe": ("ap.llm.probe_lure", "l8", ["--l3", "/tmp/l3.jsonl", "--l2", "/tmp/l2.jsonl"]),
    "boot": ("ap.llm.boot", "b1", ["--l3", "/tmp/l3.jsonl", "--scores-dir", "/tmp/scores"]),
}


def _run(study, model, dtype, batch, l3_text, l2_text, extra):
    import subprocess
    mod, _, args = STUDY[study]
    import shutil
    if os.path.exists("/tmp/out.jsonl"):          # warm containers are reused across calls
        os.remove("/tmp/out.jsonl")
    shutil.rmtree("/tmp/scores", ignore_errors=True)
    open("/tmp/l3.jsonl", "w").write(l3_text)
    open("/tmp/l2.jsonl", "w").write(l2_text)
    cmd = ["python", "-m", mod, "--model", model, "--dtype", dtype, "--out", "/tmp/out.jsonl", *args, *extra]
    if study in ("intervene", "behave", "suppress", "probe", "boot"):
        cmd += ["--batch", str(batch)]
    env = {**os.environ, "PYTHONPATH": "/root/src", "HF_HOME": "/cache"}
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    hf.commit()
    import json
    out = open("/tmp/out.jsonl").read() if os.path.exists("/tmp/out.jsonl") else ""
    out = "".join(l + "\n" for l in out.splitlines() if l.strip() and json.loads(l).get("model") == model)
    files = {}
    if os.path.isdir("/tmp/scores"):
        for fn in os.listdir("/tmp/scores"):
            files[fn] = open(os.path.join("/tmp/scores", fn), "rb").read()
    return {"model": model, "returncode": p.returncode, "out": out, "files": files,
            "log": (p.stdout + p.stderr)[-20000:]}


@app.function(gpu="H100", timeout=12 * 3600, volumes={"/cache": hf})
def run_big(*a):
    return _run(*a)


@app.function(gpu="B200", timeout=12 * 3600, volumes={"/cache": hf}, memory=65536)
def run_best(*a):
    return _run(*a)


@app.function(gpu="L40S", timeout=12 * 3600, volumes={"/cache": hf})
def run_small(*a):
    return _run(*a)


def big(model):
    return any(s in model for s in ("6.9b", "7B", "7b"))


def _toy(config, sets, n, seeds):
    """Train toy-model seeds for one N in parallel processes on one GPU; returns the result rows."""
    import subprocess
    env = {**os.environ, "PYTHONPATH": "/root/src"}
    procs = []
    for s in seeds:
        out = f"/tmp/toy_s{s}.jsonl"
        if os.path.exists(out):
            os.remove(out)
        cmd = ["python", "-m", "ap.train", f"/root/{config}", "--set", f"n={n}", "threads=1", "device=cuda",
               *sets, "--seeds", str(s), "--out", out]
        procs.append((out, subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)))
    rows, logs = "", ""
    for out, p in procs:
        logs += p.communicate()[0][-3000:]
        if os.path.exists(out):
            rows += open(out).read()
    return {"n": n, "out": rows, "log": logs}


@app.function(gpu="H100", timeout=12 * 3600, cpu=16)
def run_toy(*a):
    return _toy(*a)


@app.local_entrypoint()
def toy(tag: str, sets: str, ns: str = "1 2 3 4 5 6", seeds: str = "0-9", config: str = "configs/sweep.yaml"):
    """modal run scripts/modal_run.py::toy --tag e9-selective-512 --sets "model.family=selective model.state=512"
    One H100 container per N; the seeds of that N train in parallel processes. Output:
    results/parts/<tag>_n<N>_modal.jsonl."""
    os.chdir(ROOT)
    a, _, b = seeds.partition("-")
    seed_list = list(range(int(a), int(b) + 1)) if b else [int(a)]
    sets_l = [f"name={tag}", *sets.split()]
    calls = {int(n): run_toy.spawn(config, sets_l, int(n), seed_list) for n in ns.split()}
    for n, c in calls.items():
        try:
            r = c.get()
        except Exception as e:
            print(f"{tag} N={n}: FAILED {type(e).__name__}: {e}", flush=True)
            continue
        open(f"logs/modal_toy_{tag}_n{n}.log", "w").write(r["log"])
        with open(f"results/parts/{tag}_n{n}_modal.jsonl", "a") as f:
            f.write(r["out"])
        print(f"{tag} N={n}: {len(r['out'].splitlines())} rows", flush=True)


@app.local_entrypoint()
def main(study: str = "intervene", models: str = "", model_list: str = "configs/llm_models.txt", extra: str = "",
         gpu: str = "auto", split_ns: str = "", tag: str = ""):
    """gpu: auto (H100 for 6B+ models, L40S otherwise) | big (H100 for all) | best (B200, batch x4) | small.
    split_ns: e.g. "2 3" runs one container per N (passes --ns <n>) for each model.
    tag: output goes to results/parts/<prefix>_<tag>_<model>.jsonl (rows carry the model name, so
    duplicates across runs are removed at analysis time)."""
    import json
    os.chdir(ROOT)
    rows = [l.split() for l in open(model_list) if l.strip() and not l.startswith("#")]
    want = set(models.split()) if models else None
    l3 = "".join(open(p).read() for p in sorted(glob.glob("results/parts/l3_*.jsonl")))
    l2 = "".join(open(p).read() for p in sorted(glob.glob("results/parts/l2_*.jsonl")))
    done_l3 = {json.loads(l)["model"] for l in l3.splitlines() if l.strip()}
    jobs = []
    for m, dtype, batch in rows:
        if want is not None and m not in want:
            continue
        if study in ("intervene", "locate", "probe") and m not in done_l3:   # suppress needs no L3 heads
            print(f"skip {m}: no L3 results yet")
            continue
        for n in (split_ns.split() or [None]):
            jobs.append((m, dtype, int(batch), extra.split() + (["--ns", n] if n else [])))
    print(f"launching {study} on Modal: {len(jobs)} containers")
    pick = lambda m: (run_best if gpu == "best" else
                      run_big if gpu == "big" or (gpu == "auto" and big(m)) else run_small)
    if gpu == "best":
        jobs = [(m, d, b * 4, ex) for m, d, b, ex in jobs]
    calls = [pick(m).spawn(study, m, d, b, l3, l2, ex) for m, d, b, ex in jobs]
    prefix = STUDY[study][1] + (f"_{tag}" if tag else "")
    os.makedirs("results/parts", exist_ok=True); os.makedirs("logs", exist_ok=True)
    pending = dict(enumerate(calls))
    while pending:                                   # collect in completion order, not launch order
        for i, c in list(pending.items()):
            try:
                r = c.get(timeout=0)
            except (TimeoutError, modal.exception.TimeoutError):
                continue
            except Exception as e:                   # container error: report and move on
                print(f"{jobs[i][0]} {jobs[i][3]}: FAILED {type(e).__name__}: {e}", flush=True)
                del pending[i]
                continue
            del pending[i]
            mtag = r["model"].replace("/", "_")
            ntag = "_".join(jobs[i][3][-1:]) if split_ns else ""
            open(f"logs/modal_{study}{'_' + tag if tag else ''}_{mtag}{'_n' + ntag if ntag else ''}.log", "w").write(r["log"])
            if r["out"]:
                with open(f"results/parts/{prefix}_{mtag}.jsonl", "a") as f:
                    f.write(r["out"])
            for fn, data in r.get("files", {}).items():
                os.makedirs(f"results/scores/{study}", exist_ok=True)
                open(f"results/scores/{study}/{fn}", "wb").write(data)
            print(f"{r['model']} {' '.join(jobs[i][3])}: exit {r['returncode']}, {len(r['out'].splitlines())} rows", flush=True)
        if pending:
            __import__("time").sleep(20)
