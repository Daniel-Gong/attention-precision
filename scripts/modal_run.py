"""Run LLM studies (L5 intervene, L7 locate, L2/L3 if needed) on Modal GPUs, one container per model.

Run from the repo root on a machine with a Modal token (Misha login node):
  modal run scripts/modal_run.py --study intervene               # every model with L3 results
  modal run scripts/modal_run.py --study locate --models "gpt2 EleutherAI/pythia-410m"

Inputs are read locally (results/parts/l3_*.jsonl, l2_*.jsonl) and passed to the container;
each container's output JSONL is written back to results/parts/<prefix>_<model>.jsonl and its
log to logs/modal_<study>_<model>.log. Hugging Face downloads persist in the Modal volume
"ap-hf-cache". Models of 6B parameters or more get an H100, the rest an L40S.
"""
import glob
import os
import pathlib

import modal

ROOT = pathlib.Path(__file__).resolve().parent.parent
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.11.0", "transformers==5.19.0", "numpy", "scipy", "pyyaml", "accelerate")
         .add_local_dir(str(ROOT / "src"), "/root/src"))
app = modal.App("attention-precision", image=image)
hf = modal.Volume.from_name("ap-hf-cache", create_if_missing=True)

STUDY = {  # study -> (module, output prefix, extra args)
    "intervene": ("ap.llm.intervene", "l5", ["--l3", "/tmp/l3.jsonl"]),
    "locate": ("ap.llm.locate", "l7", ["--l3", "/tmp/l3.jsonl", "--l2", "/tmp/l2.jsonl", "--ns", "2", "3", "4"]),
    "heads": ("ap.llm.heads", "l3", ["--ns", "1", "2", "3", "4"]),
    "behave": ("ap.llm.behave", "l2", []),
}


def _run(study, model, dtype, batch, l3_text, l2_text, extra):
    import subprocess
    mod, _, args = STUDY[study]
    open("/tmp/l3.jsonl", "w").write(l3_text)
    open("/tmp/l2.jsonl", "w").write(l2_text)
    cmd = ["python", "-m", mod, "--model", model, "--dtype", dtype, "--out", "/tmp/out.jsonl", *args, *extra]
    if study in ("intervene", "behave"):
        cmd += ["--batch", str(batch)]
    env = {**os.environ, "PYTHONPATH": "/root/src", "HF_HOME": "/cache"}
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    hf.commit()
    out = open("/tmp/out.jsonl").read() if os.path.exists("/tmp/out.jsonl") else ""
    return {"model": model, "returncode": p.returncode, "out": out, "log": (p.stdout + p.stderr)[-20000:]}


@app.function(gpu="H100", timeout=12 * 3600, volumes={"/cache": hf})
def run_big(*a):
    return _run(*a)


@app.function(gpu="L40S", timeout=12 * 3600, volumes={"/cache": hf})
def run_small(*a):
    return _run(*a)


def big(model):
    return any(s in model for s in ("6.9b", "7B", "7b"))


@app.local_entrypoint()
def main(study: str = "intervene", models: str = "", model_list: str = "configs/llm_models.txt", extra: str = ""):
    os.chdir(ROOT)
    rows = [l.split() for l in open(model_list) if l.strip() and not l.startswith("#")]
    want = set(models.split()) if models else None
    l3 = "".join(open(p).read() for p in sorted(glob.glob("results/parts/l3_*.jsonl")))
    l2 = "".join(open(p).read() for p in sorted(glob.glob("results/parts/l2_*.jsonl")))
    done_l3 = {__import__("json").loads(l)["model"] for l in l3.splitlines() if l.strip()}
    jobs = []
    for m, dtype, batch in rows:
        if want is not None and m not in want:
            continue
        if study in ("intervene", "locate") and m not in done_l3:
            print(f"skip {m}: no L3 results yet")
            continue
        jobs.append((m, dtype, int(batch)))
    print(f"launching {study} on Modal for {len(jobs)} models")
    calls = [(run_big if big(m) else run_small).spawn(study, m, d, b, l3, l2, extra.split()) for m, d, b in jobs]
    prefix = STUDY[study][1]
    os.makedirs("results/parts", exist_ok=True); os.makedirs("logs", exist_ok=True)
    for c in calls:
        r = c.get()
        tag = r["model"].replace("/", "_")
        open(f"logs/modal_{study}_{tag}.log", "w").write(r["log"])
        if r["out"]:
            with open(f"results/parts/{prefix}_{tag}.jsonl", "a") as f:
                f.write(r["out"])
        print(f"{r['model']}: exit {r['returncode']}, {len(r['out'].splitlines())} rows", flush=True)
