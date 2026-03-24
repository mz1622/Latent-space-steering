# AGENTS.md — Codex Governance & Workflow

This file is the single source of truth for Codex working in this repository.

Scope note: All work for this project is done under `Latent-space-steering/SAE/`. Documentation coordination files live in `Latent-space-steering/SAE/docs/` (not the repo root docs directory).

## Your role
You are a senior engineer. Your job is to implement user requests with:
- minimal, surgical changes
- correctness and reproducibility
- clear structure and documentation

Avoid “rewrite everything.” Refactor only as needed for the task.

## Hard rules
1) Minimal change
- Touch the fewest files possible.
- Avoid stylistic rewrites unless they remove duplication or fix correctness.

2) Keep the repo compiling
After edits, ensure no syntax/import errors.
Preferred checks (run what exists):
- `python -m py_compile ...` (at least changed files)
- `pytest -q` if tests exist
- `ruff` / `black --check` if configured
If execution isn’t available, do a careful reasoning-based compile check:
- imports resolve
- CLI entry points exist
- paths are correct

3) Debug after editing
If you introduce a bug, fix it in the same task. Do not leave the repo broken.

4) Delete dead code
If a file/function is unused, remove it AFTER confirming it has no callers.
Prefer deleting whole dead modules over leaving abandoned utilities.

5) No silent behavior changes
If you change:
- CLI flags/defaults
- output paths
- file naming/layout
Then you MUST update README and log it in HISTORY (and DEVLOG if major).

6) No duplication
If you are about to copy/paste > ~20 lines, stop and create/reuse a shared helper.

7) Header comment required for touched Python files
At the top of every modified/created Python file, add a short header:
- what the file does
- how it is used (entry points / main functions)
- key inputs/outputs

## AWS / Remote HPC special notes (must follow on EC2)
These rules exist because EC2 typically has a small root disk and a large attached data disk (e.g., 30GB `/` + 500GB `/mnt`). Most failures come from writing large files/caches to `/`.

1) Disk layout rule
- Treat `/` as system-only. Do NOT download datasets or models to `/home` by default.
- Put all large artifacts on `/mnt` (or the designated large volume mount):
  - repos/workspace: `/mnt/work/<repo>`
  - datasets: `/mnt/datasets/...`
  - models: `/mnt/models/...` (optional; Hub IDs are also fine)
  - caches: `/mnt/.cache/...`
  - tmp: `/mnt/tmp`
- Before any large download, check free space:
  - `df -h / /mnt`
  - If `/` is getting tight, migrate to `/mnt` before continuing.

2) Hugging Face / Transformers cache must be on `/mnt`
When running on EC2, set these env vars (in shell and `~/.bashrc`) so downloads never go to `~/.cache` on the root disk:
- `HF_HOME=/mnt/.cache/huggingface`
- `TRANSFORMERS_CACHE=/mnt/.cache/huggingface`
- `TORCH_HOME=/mnt/.cache/torch`
- `TMPDIR=/mnt/tmp`
Also consider:
- `PIP_CACHE_DIR=/mnt/.cache/pip`

3) Never use Windows paths on Linux EC2
- Any config path like `D:/...` is invalid on EC2. Use Linux paths (`/mnt/...`) or Hub repo IDs (`namespace/repo_name`).
- Maintain separate configs for local vs EC2 if needed (e.g., `configs/local.yaml` vs `configs/ec2.yaml`), and document which one to use.

4) Confirm you are on the right machine before running heavy jobs
- When using VS Code Remote-SSH, verify you are in the remote window.
- In shell, sanity check:
  - `hostname`
  - `pwd`
  - `nvidia-smi` (confirms GPU node and driver)

5) Long-running jobs must be resilient
- Use `tmux` for training/evaluation jobs so SSH disconnects don’t kill runs.
- Write outputs/checkpoints to `/mnt` (and optionally sync to S3).

6) Volume mounts are fragile; don’t break them
- If editing `/etc/fstab`, run `sudo mount -a` to validate formatting.
- Avoid changing mountpoints used by code without updating docs/configs.

## Required project md files (`Latent-space-steering/SAE/docs/`)
These files coordinate agent work. Create any missing ones under `Latent-space-steering/SAE/docs/`.

- `README.md`        : user-facing usage + structure for SAE workstream
- `Current_task.md`  : active plan/checklist for current work (scratchpad)
- `MEMORY.md`        : compact long-term memory for agents (keep short)
- `HISTORY.md`       : detailed action log (append-only; can compact redundancy)
- `DEVLOG.md`        : major milestones only (timestamped)
- `CLAUDE.md`        : for claude code, no need to read
- method docs (`METHOD.md`) under `Latent-space-steering/SAE/src/methods/<method>/METHOD.md`:
  Each must include:
  - math explanation (high level but correct)
  - code structure (what files/classes/functions matter)
  - commands to run fit/eval with the new scripts
  - typical inputs/outputs and where outputs are saved (include EC2 `/mnt` notes if relevant)

## Workflow (must follow)
When the user gives a task:

1) Read
- Read:
  - `Latent-space-steering/SAE/README.md` (if present)
  - `Latent-space-steering/SAE/docs/DEVLOG.md`
  - `Latent-space-steering/SAE/docs/HISTORY.md`
  - `Latent-space-steering/SAE/docs/Current_task.md`
- If task touches methods: read corresponding `Latent-space-steering/SAE/src/methods/<method>/METHOD.md` (or create/update at end)

2) If Current_task is empty or not aligned with user prompt, plan
- Write a concrete plan in `Latent-space-steering/SAE/docs/Current_task.md`:
  - goal/scope
  - files to inspect
  - expected output paths (include EC2 `/mnt` paths if relevant)
  - step-by-step todo checklist
  - acceptance criteria
  Otherwise follow `Current_task.md`.

3) Implement incrementally
- Make coherent small edits.
- Keep imports/entry points working at each step.
- Consolidate duplicated logic.

4) Validate
- Run quick checks (or best-effort compile sanity).
- Fix errors immediately.
- On EC2, validate disk/caches for any workflow that downloads models/data.

5) Document & close
- Append detailed changes to `Latent-space-steering/SAE/docs/HISTORY.md` (paths + why). Add these changes to the top of the file.
- If major: add a concise entry to `Latent-space-steering/SAE/docs/DEVLOG.md`:
  - timestamp (local)
  - what changed
  - where it lives (paths)
  - minimal run commands
- Update SAE README if anything user-facing changed.
- If method behavior/usage changed, update `METHOD.md`.
- Clear `Current_task.md` and distill outcome into `MEMORY.md` in simplest useful form:
  - user issue
  - key edits
  - solution summary
  - caveats / follow-ups (include EC2-specific caveats if applicable)

## Coding preferences
- Prefer explicit code and simple registries over complex abstractions.
- Centralize common logic (CLI parsing, output dir, benchmark loading) to avoid drift.
- Keep methods isolated under SAE methods/ (methods should not own dataset logic).

## Output management
- Write outputs under a single `outputs/` root within `Latent-space-steering/SAE/` (or a configured output root).
- Do not commit generated outputs.
- Document output layout in SAE README.
- On EC2: outputs, logs, checkpoints must be on `/mnt` (or the large mounted volume), not on `/`.

## Deletions/renames
Before deleting/renaming:
- search for references (ripgrep)
- update import paths
- update docs