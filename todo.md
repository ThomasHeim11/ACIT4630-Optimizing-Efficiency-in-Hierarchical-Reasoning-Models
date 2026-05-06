# SHREK-HRM Submission TODO

Complete pre-submission cleanup list. Covers `main.tex`, code, FLOPs, and submission package.

Updated based on the latest `main.tex` revision.

---

## 🔴 1. main.tex — must-fix factual errors and placeholders

- [x] **Fill in word count** in title — placeholder removed
- [x] **Remove Norwegian TODO comment** in Introduction
- [x] **Fix wrong citations** in Materials section — `[hoffmann2022training, cuskley2024limitations]` no longer appears for HRM/TRM
- [x] **Fix "six key findings" miscount** — now says "five key findings" matching 5 items
- [x] **Fix "augmented hint variant"** — no longer mentions the hint variant
- [x] **Fix sample-size inconsistency** — now consistently 1{,}000
- [x] **Fix "All four models"** — now correctly says four (HRM, TRM, SHREK-Large, SHREK-Small)
- [ ] **Fix Sudoku hyperparameters in Methodology** — still says `learning rate $1 \times 10^{-4}$ (or $7 \times 10^{-5}$ for Sudoku), and global batch size 384`. Your verified Sudoku runs used **lr=1e-4, batch=768** (matching TRM); change accordingly - Max
- [x] **Remove ARC-AGI mentions** — done
- [x] **Capitalize subsection titles** — `\subsection{Limitations}` and `\subsection{Future Work}` now properly capitalized

---

## 🔴 2. main.tex — naming consistency

- [x] **Unify "SHREK Tiny" vs "SHREK Small"** — consistently SHREK-Small everywhere
- [x] **TRM MLP consistency** — removed from both Sudoku table and FLOPs table

---

## 🗑️ 3. main.tex — REMOVE redundancies

### Direct deletions

- [x] **Delete annotation block** after `\end{document}`
- [x] **Delete intro repeat** — line 73 duplicate of line 62 removed
- [x] **Delete duplicated paragraph** — "We removed the second inner forward pass..." now appears only once

### Drop benchmark literature subsections

- [x] **Delete subsection** *"Can We Trust AI Benchmarks?"*
- [x] **Delete subsection** *"Assessing Small Language Models for Code Generation"*
- [x] **Delete subsection** *"Benchmarking is Broken"*
- [x] **Optional**: fold one sentence into Introduction citing Eriksson — not done; OK to skip

### Compress overlapping sections

- [ ] **Cut Discussion section** — still has 5 paragraphs that overlap heavily with Conclusion (error injection is biggest contributor, SHREK-Small competitive, Sudoku halts earlier, Maze costs more). Cut to 2–3 paragraphs of fresh analysis.
- [ ] **Merge or compress** Section IV-C (Methodology) ↔ Section IV-D (Proposed Solution) — both still describe the architecture
- [ ] **Compress Implementation Plan** — Section III still has 4 subsections (Research Gaps, Research Objective, Research Question, Problem Statement). Consider folding into Background + opening of Section IV per the IEEE-recommended structure

---

## 🗑️ 4. Repository cleanup — REMOVE files/folders

- [ ] **Remove empty `build/` folder** at repo root: `rmdir build/`
- [ ] **Remove stale FLOPs JSONs** if any reappear in `flops/results/`:
  - `shrek_tiny_sudoku.json`, `shrek_tiny_maze.json` (renamed → small)
  - `trm_mlp_sudoku.json` (TRM MLP dropped from comparison)
  - `augmented_hrm_sudoku.json` (Augmented HRM not in this paper)
- [ ] **Remove old `models/` folder on cluster** after confirming everything moved to `source/`: `rm -rf ~/HMR/models` (only after `find ~/HMR/models -type f` shows nothing important)
- [ ] **Check for `.bak` files** from earlier sed operations: `find . -name "*.bak"` then delete
- [ ] **Decide whether to ship** development helpers (not needed by grader):
  - `verify_local.sh` (dev convenience)
  - `testLocal.py` (dev convenience)
  - `run_test.sh` (cluster-only SLURM wrapper)
  - `cluster.md` (your private notes about cluster login)
  - `SHREKV4.md` (V4 work plan, not relevant to V1 paper)
  - These can be excluded from `Project-Attachment-GroupXX.zip` even if kept in repo

---

## ✏️ 5. Spelling / grammar pass

### Already fixed
- [x] "hierarchial" → "hierarchical"
- [x] "ocisilation" → "oscillation"
- [x] "reasonin" → "reasoning"
- [x] "compational" → "computational"
- [x] "consistenyl" → "consistently"
- [x] "despiste" → "despite"
- [x] "primarliy" → "primarily"
- [x] "Shrek LARGE" → "SHREK-Large"
- [x] "MAZE" → "Maze-Hard"
- [x] "sudoku" → "Sudoku-Extreme"
- [x] "form" → "from"
- [x] "effecting" → "effective"

### Still to fix (NEW typos found in current revision)
- [X] **"Suskoko"** → "Sudoku" — Summary of Findings, item 3: *"+10\% over Original HRM in ablation on Suskoko dataset"*
- [X] **"fip rate"** → "flip-rate" — Conclusion: *"By combining the fip rate signals with a learned error estimator"*
- [X] **"Sudoku-Extremedataset"** missing space — Ablation Study: *"using the Sudoku-Extremedataset"*
- [X] **"stabilize effect"** → "stabilizing effect" — Ablation Study: *"control for its known stabilize effect"*
- [X] **"green ai"** → "Green AI" — Conclusion: *"This aligns with green ai \cite{schwartz2019greenai}"*
- [X] **"more computing"** → "more compute" — Conclusion: *"it might also require more computing"*
- [X] **"Sudoku-Extreme(32"** missing space — Methodology: *"global batch size 384 for Sudoku-Extreme(32 for smoke tests)"*

---

## 📊 6. FLOPs tables — update with latest measurements

- [x] FLOPs JSONs are current (latest cluster run completed)
- [x] Plots regenerated locally with correct 4-models-per-task structure (no SHREK Tiny phantom, no TRM MLP)
- [ ] Verify Table V (Sudoku FLOPs) numbers in main.tex match `flops/results/*_sudoku.json`
- [ ] Verify Table VI (Maze FLOPs) numbers in main.tex match `flops/results/*_maze.json`
- [ ] Confirm `flops/sudoku_accuracy_vs_flops.png` and `flops/maze_accuracy_vs_flops.png` are the regenerated versions in the report

---

## ⚠️ 7. main.tex — content questions to verify

- [ ] **Ablation Table IV "HRM + EMA" 57%** — chart we verified shows ~67% peak. Either re-verify the source, add a footnote, or replace with ~67%
- [x] **Section II-A "apples to apples"** — wording rephrased
- [x] **Section III "Gap 2: Gap 2"** duplicated label — fixed
- [ ] **Section IV-A "Approach"** — bullet list still mentions early plan terms (*"Internal Logical Floor"*, *"Inference-Time Stuck Alarm"*) but Section IV-D uses *"error injection"* and *"stagnation delta"* — unify terminology
- [ ] **Section IV-A caveat paragraph** — *"We were not able to generate enough computational resources to train these models on our own"* contradicts your later statement that you DID train all models on the cluster. Remove this paragraph or rephrase

---

## 🟡 8. Structural restructuring (optional but recommended)

The teacher's spec prescribes: **Introduction → Background → Proposed Solution → Experiments → Results → Discussion → Conclusion**.

Current Section III (*Research Gaps, Research Problem, Research Question and Problem Statement*) doesn't fit. Suggested restructure:

- [ ] **Move "Research Gaps"** to end of Section II (Background) as final subsection
- [ ] **Compress "Research Objective + Research Question"** into 2 sentences at the end of the Introduction
- [ ] **Compress "Problem Statement"** into 1 short paragraph at the start of new Section III (Proposed Solution)
- [ ] **Delete Section III header**, renumber subsequent sections
- [ ] Section IV (Method and Materials) → split into separate **Section III: Proposed Solution** and **Section IV: Experiments** to match teacher's structure

---

## 📦 9. Submission package — final assembly

- [ ] Add `huggingface_hub` to `requirements.txt`
- [ ] Add `pyyaml` to `requirements.txt`
- [ ] Verify `requirements.txt` contains all needed: `torch`, `flash-attn`, `omegaconf`, `pydantic`, `hydra-core`, `einops`, `tqdm`, `wandb`, `adam-atan2-pytorch`, `huggingface_hub`, `pyyaml`
- [ ] Update `README.md` to include:
  - Setup instructions (`pip install -r requirements.txt`)
  - Hardware requirement (NVIDIA GPU + CUDA 12.6)
  - How to run `python test.py`
  - Expected output table matching Tables I + II
  - Note about HuggingFace download on first run (~16GB)
- [ ] Final test of `test.py` on cluster — confirm all 8 rows produce non-`n/a` accuracy values
- [ ] Verify `source/` directory contains code (per teacher's required structure)
- [ ] Verify `model/` directory is auto-created on first `test.py` run (or include link to HuggingFace in README)
- [ ] Verify `data/` directory is auto-created on first `test.py` run (or include link to HuggingFace in README)
- [ ] Compile `main.tex` → `Project-Report-GroupXX.pdf`
- [ ] Build `Project-Attachment-GroupXX.zip` containing:
  - `source/`
  - `requirements.txt`
  - `README.md`
  - `test.py`
  - `config.yaml`
  - LaTeX source files (`main.tex`, `bib.bib`, any included figures)
- [ ] Submit `Project-Report-GroupXX.pdf` and `Project-Attachment-GroupXX.zip`
- [ ] **Update Individual contributions appendix** — currently all 100% which is unusual; teacher expects percentages relative to the most contributing person being 100%. Some members may end up at 80–90%

---

## 🟢 10. Things that are GOOD — DO NOT change

- Introduction's logical flow (LLM limits → Green AI → HRM → TRM → SHREK)
- Ablation Table IV structure (component breakdown is clear)
- FLOPs analysis (Tables V/VI + accuracy-vs-FLOPs figures) — strongest paper contribution
- Honest limitations section
- Future Work section (concrete directions)
- Math notation (flip rate equation, error injection formula, total loss)

---

## 🎯 Remaining priority order

1. **Sudoku hyperparameters fix** (Section 1) — only must-fix factual error left
2. **New typos** (Section 5: Suskoko, fip rate, Sudoku-Extremedataset, stabilize, green ai, more computing)
3. **Approach terminology drift** (Section 7) — Logical Floor / Stuck Alarm unify
4. **Caveat paragraph contradiction** (Section 7) — remove or rephrase
5. **Cut Discussion redundancy** (Section 3)
6. **Verify Ablation Table IV 57%** (Section 7)
7. **Optional**: Section III restructure (Section 8)
8. **FLOPs table number verification** (Section 6)
9. **Repo cleanup** (Section 4)
10. **README + requirements.txt + test.py verification** (Section 9)
11. **Build ZIP and submit** (Section 9)
