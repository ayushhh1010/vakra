# Setup on a Windows laptop (RTX 3050 4 GB, 16 GB RAM)

Everything here runs natively on Windows in PowerShell. Day 1 needs no Docker and almost no disk. Day 2 adds Docker for VAKRA's other capabilities.

Folder layout used throughout:

```
C:\Users\<you>\vakra        IBM's repository (git clone https://github.com/IBM/vakra)
C:\Users\<you>\vakra-rl     this repository
```

If you already have a `vakra` folder, check it is IBM's clone: `git -C $HOME\vakra remote -v` should show `github.com/IBM/vakra`. Update it with `git -C $HOME\vakra pull`.

---

## Day 1: first real baseline, no Docker

### 1. Python environment

```powershell
cd $HOME\vakra-rl
py -3.11 -m venv .venv            # any Python 3.11 or newer
.\.venv\Scripts\Activate.ps1      # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install -U pip
pip install -r requirements.txt
pip install -e .
python -m pytest -q               # 8 tests should pass
```

### 2. Model server (Ollama)

```powershell
ollama pull qwen3:4b-instruct-2507-q4_K_M     # 2.5 GB; the instruct model does not "think", which keeps tool calls clean
```

Ollama's default context window can be smaller than an agent prompt with 30+ tool schemas, and it truncates silently. Raise it, then **quit Ollama from the system tray and start it again**:

```powershell
setx OLLAMA_CONTEXT_LENGTH 8192
```

Check the model loads and how much sits on the GPU:

```powershell
ollama run qwen3:4b-instruct-2507-q4_K_M "Say hi"
ollama ps          # PROCESSOR column: "100% GPU" is ideal; a CPU/GPU split is slower but works
```

### 3. BPO baseline

BPO (recruiting analytics) is the one VAKRA domain whose tool server and tasks live inside the repository: 32 tools, 45 tasks.

```powershell
python scripts\run_bpo_baseline.py --vakra-root ..\vakra --limit 3     # smoke test, about a minute
python scripts\run_bpo_baseline.py --vakra-root ..\vakra               # all 45 tasks
```

Open `runs\<time>_bpo_<model>\report.md`. Then measure reliability: same tasks, three attempts each, with sampling on.

```powershell
python scripts\run_bpo_baseline.py --vakra-root ..\vakra --repeats 3 --temperature 0.7
```

`pass^3` (all three attempts right) against `pass@1` tells you how consistent the model is. That gap is one of the things RL should shrink.

Optional reference point with a large model (free Groq key from console.groq.com):

```powershell
$env:GROQ_API_KEY = "gsk_..."
python scripts\run_bpo_baseline.py --vakra-root ..\vakra --provider groq --model openai/gpt-oss-120b
```

### 4. Training-split data and analysis

```powershell
python scripts\download_vakra.py --parts train --max-file-mb 50          # prints the plan and sizes only
python scripts\download_vakra.py --parts train --max-file-mb 50 --yes    # roughly 200 MB
python scripts\analyze_data.py --data-root ..\vakra\data
```

`--max-file-mb 50` skips a handful of huge ground-truth files (one capability-2 file alone is 2.2 GB). The report lists exactly what was skipped. Results land in `reports\data\`.

---

## Day 2: VAKRA capabilities through Docker

### 5. Docker Desktop

Install Docker Desktop with the WSL 2 backend. In Settings > Resources, give it 6 GB of memory (8 GB only if you run capability 4, which needs ChromaDB). Check: `docker version`.

### 6. Build VAKRA's image

```powershell
cd $HOME\vakra
docker build -t benchmark_environ -f docker/Dockerfile.unified .
```

This image installs sentence-transformers and its PyTorch dependency, so expect several GB. Check `docker images` afterwards and keep an eye on free disk space.

### 7. Databases for a few small domains

The tool servers read per-domain SQLite databases. Six domains account for about 19 GB; most others are tiny.

```powershell
cd $HOME\vakra-rl
python scripts\download_vakra.py --list-db-sizes
python scripts\download_vakra.py --parts databases --domains beer_factory disney computer_student cars mondial_geo books hockey
python scripts\download_vakra.py --parts databases --domains beer_factory disney computer_student cars mondial_geo books hockey --yes
```

These seven all appear in the capability-3 training split and are each under 10 MB.

### 8. Start one container and run

```powershell
cd $HOME\vakra
docker compose up -d capability_3_multihop_reasoning
docker compose ps
Start-Sleep 60                                   # the internal API needs up to a minute (VAKRA's own script waits 120 s)
docker logs --tail 20 capability_3_multihop_reasoning

cd $HOME\vakra-rl
python scripts\run_vakra_baseline.py --capability 3 --domain beer_factory disney --limit 10
```

The run folder contains `report.md`, `trajectories.jsonl` and `prediction\<domain>.json` in VAKRA's submission format.

### 9. Official VAKRA score (optional, uses an LLM judge)

IBM's evaluator replays the predicted tool calls in the containers and asks `openai/gpt-oss-120b` to judge the answers. Its default backend is IBM-internal, so point it at Groq. Use a separate environment inside the VAKRA folder, following VAKRA's `setup.md`:

```powershell
cd $HOME\vakra
py -3.11 -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -e ".[init]"; pip install -r requirements_benchmark.txt
$env:JUDGE_BACKEND = "groq"; $env:API_KEY = "gsk_..."
python evaluator\evaluator.py --capability_name capability_multihop_reasoning `
  --gt_root data\train\capability_3_multihop_reasoning\output `
  --pred_root ..\vakra-rl\runs\<run>\prediction `
  --domains beer_factory disney `
  --output ..\vakra-rl\runs\<run>\official_results.json
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Could not start the MCP tool server` | Run the command printed in the error by hand to see the real error, usually a missing package (`pip install -r requirements.txt`). |
| Many `format_error` or `Tool calls recovered from plain text` | The model writes tool calls as text. Confirm you pulled the `instruct-2507` tag and that the context length was raised. |
| Answers ignore obvious tool results | Prompt may be truncated. Check "Largest prompt seen" in the report against `OLLAMA_CONTEXT_LENGTH`. |
| Very slow (minutes per task) | `ollama ps` shows heavy CPU offload. Close other GPU apps, or use `--limit` while developing. |
| `UnicodeDecodeError` / `charmap` codec errors | `setx PYTHONUTF8 1` and open a new terminal. |
| Capability 4 container exits at start | Docker memory below 8 GB (ChromaDB). |
