# SHREK-HRM

This repository accompanies the ACIT4630 group project report on SHREK-HRM (Self-Corrective Hierarchical Reasoning Model). `test.py` reproduces the SHREK-Large and SHREK-Small evaluation numbers reported in the paper.

---

## Prerequisites

- **Simula Research Laboratory cluster** (required, the code depends on `flash-attn` and `adam-atan2` which require Linux and NVIDIA CUDA. It will not run on Windows or macOS.)
- NVIDIA GPU with CUDA 12.6 (tested on GH200 via the `gh200q` partition)
- Python 3.10+
- ~20 GB free disk for the auto-downloaded checkpoints + datasets

## Setup

`flash-attn` must be compiled from source and requires CUDA. Install in two steps:

```bash
module load cuda12.6/toolkit/12.6.3
pip install -r requirements.txt
pip install flash-attn==2.8.3 --no-build-isolation
```

## Run (Simula cluster)

The model was trained on Simula's `gh200q` partition; `run_test.sh` is a portable SLURM wrapper that runs `test.py` on the same partition. Logs go into `./logs/` relative to the directory you submit from — no paths are hardcoded.

#### 1. Copy the submission ZIP to the cluster

From your local machine:

```bash
scp Project-Attachment-Group03.zip <user>@dnat.simula.no:~/
```

#### 2. SSH into the cluster

```bash
ssh <user>@dnat.simula.no -p 60441
```

#### 3. Unzip and enter the project

```bash
unzip Project-Attachment-Group03.zip -d shrek-hrm
cd shrek-hrm
```

#### 4. Install dependencies

```bash
module load cuda12.6/toolkit/12.6.3
pip install --upgrade pip
pip install -r requirements.txt
pip install flash-attn==2.8.3 --no-build-isolation
```

The `flash-attn` install takes ~10 minutes as it compiles from source. It must be installed separately because it requires `torch` to be present during build.

#### 5. Submit the evaluation job

```bash
mkdir -p logs
sbatch run_test.sh
```

`sbatch` prints a job ID — note it down.

#### 6. Wait for the job to finish

```bash
squeue -u $USER
```

Once the job disappears from the queue (typically 10–15 min after it starts running), it is done. Queue wait time depends on cluster load.

#### 7. Read the results

```bash
cat logs/test_py_<JOBID>.log
```

The script prints an accuracy table with one row per (model, task) pair, matching the format in the report.

### Notes

- On first run, `test.py` automatically downloads the checkpoints and test datasets from HuggingFace into `model/` and `data/`. Subsequent runs skip the download.
- All paths and HuggingFace repo IDs live in `config.yaml` so `test.py` itself runs unmodified. Edit `config.yaml` only if you need to point at a different checkpoint or dataset location.
- Total runtime is ~10–15 min on a single NVIDIA GH200 (or comparable GPU).

---

## Training

Training scripts are provided in `source/SHREK-HRM/script/train/`. Each script is a SLURM job configured for the Simula `gh200q` partition. To train SHREK-Large on Maze-Hard, for example:

```bash
cd source/SHREK-HRM
sbatch script/train/train_shrek_large_maze.sh
```

Available training scripts:

| Script                      | Model       | Dataset   |
| --------------------------- | ----------- | --------- |
| `train_shrek_large_maze.sh` | SHREK-Large | Maze-Hard |
| `train_shrek_tiny_maze.sh`  | SHREK-Small | Maze-Hard |

Ablation study scripts are in `source/SHREK-HRM/script/train/AblationStudy/` and cover all configurations reported in Table IV of the paper. All hyperparameters are set within the scripts (learning rate, batch size, epochs, etc.) and match the values reported in the paper. Training requires a single NVIDIA GH200 GPU and the datasets from `dataset/data/`.

---

## Repository layout

```
shrek-hrm/
├── source/             # Model code (SHREK-HRM, HRM, TRM)
├── flops/              # FLOPs measurement scripts
├── model/              # Auto-created on first test.py run (HuggingFace download)
├── data/               # Auto-created on first test.py run (HuggingFace download)
├── config.yaml         # Paths + settings consumed by test.py
├── test.py             # Evaluation entry point
├── run_test.sh         # SLURM wrapper for the Simula cluster
├── requirements.txt    # Pip dependencies
```

---

## Pre-trained artefacts

- **Checkpoints**: https://huggingface.co/ThomasHeim/HRM-Reproduction-Checkpoints
- **Dataset**: https://huggingface.co/datasets/ThomasHeim/HRM-dataset
- **Source code**: https://github.com/ThomasHeim11/SHREK-HRM

Both are downloaded automatically by `test.py` — no manual setup required.
