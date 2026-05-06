# SHREK-HRM Submission TODO

Complete pre-submission cleanup list. Covers `main.tex`, code, FLOPs, and submission package.

---

## 🔴 1. main.tex — must-fix factual errors and placeholders

- [X] **Fill in word count** in title (line 20): replace `Wordcount: [Insert Count]` with the actual count
- [X] **Remove Norwegian TODO comment** in Introduction (line 48): `% Tror det kan være greit å prøve å svare på...`
- [ ] **Fix wrong citations** in Materials section (line 236): `\cite{hoffmann2022training, cuskley2024limitations}` is wrong for HRM/TRM reference numbers — change to `\cite{wang2025hierarchicalreasoningmodel, jolicoeurmartineau2025morerecursivereasoningtiny}`
- [X] **Fix "six key findings" miscount** (line 451) — list contains only 5 items, either add a 6th or change "six" → "five"
- [X] **Fix "augmented hint variant"** (line 230) — your verified runs use **vanilla** Sudoku-Extreme-1k-aug-1000 (no hint), not the hint variant
- [x] **Fix sample-size inconsistency** — line 394 says "100 test puzzles", line 305 says "1000 test samples"; pick one and use everywhere
- [X] **Fix "All four models"** (line 230) — you trained 5: HRM, TRM-Att, TRM-MLP, SHREK-Large, SHREK-Tiny
- [ ] **Fix Sudoku hyperparameters in Methodology** (line 230) — currently says `lr=7e-5, batch=384`, but your verified runs used `lr=1e-4, batch=768` (matching TRM)
- [X] **Remove ARC-AGI mentions** in Methodology (line 232) — you only evaluate Sudoku + Maze; either drop ARC-AGI or move to Future Work consistently
- [X] **Capitalize subsection titles** for consistency: `\subsection{limitations}` → `\subsection{Limitations}`, same for `\subsection{future work}` → `\subsection{Future Work}`

---

## 🔴 2. main.tex — naming consistency

- [X] **Unify "SHREK Tiny" vs "SHREK Small"** — FLOPs script renamed Tiny → Small. Decide on one name, then update all of:
  - Tables II, III, IV (results)
  - Tables V, VI (FLOPs)
  - Discussion paragraphs
  - Conclusion paragraphs
  - Abstract / contributions list
- [X] **TRM MLP consistency** — appears in Table II (Sudoku, 84%) and FLOPs Sudoku table, but you decided to drop TRM MLP from FLOPs comparison. Either:
  - Remove TRM MLP row from both Table II and FLOPs table, OR
  - Keep TRM MLP in both with a note (e.g., "MLP variant shown for reference; not included in FLOPs comparison because it uses a different architecture family")

---

## 🗑️ 3. main.tex — REMOVE redundancies

### Direct deletions

- [X] **Delete annotation block** lines 521–580 (everything after `\end{document}`) — leftover chat notes that shouldn't ship with the source
- [ ] **Delete intro repeat** (line 73) — *"Motivated by this insight, we propose the SHREK-HRM..."* duplicates the same sentence in line 62
- [ ] **Delete duplicated paragraph** lines 272–276 — the two sentences *"We removed the second inner forward pass that AugmentedHRM used... two variants..."* appear twice in Section IV-D

### Drop benchmark literature subsections

None of these inform SHREK's design. Removing tightens lit review by ~1.5 PDF pages.

- [X] **Delete subsection** *"Can We Trust AI Benchmarks?"* (Eriksson et al.) — line 118
- [X] **Delete subsection** *"Assessing Small Language Models for Code Generation"* (Hasan et al.) — line 126
- [X] **Delete subsection** *"Benchmarking is Broken"* (Cheng et al.) — line 134
- [ ] **Optional**: fold one sentence into Introduction citing Eriksson — *"Concerns about benchmark trustworthiness [eriksson2025trustbenchmarks] further motivate honest, identical-hardware comparisons of compact reasoning models."*

### Compress overlapping sections

- [ ] **Cut Discussion section** (Section VI) to ~2 short analytical paragraphs — currently 4 paragraphs that overlap heavily with Conclusion; both say:
  - Error injection is the biggest contributor
  - SHREK Tiny matches HRM with fewer parameters
  - Sudoku halts earlier saves compute
  - Maze requires more steps for higher accuracy
- [ ] **Merge or compress** Section IV-C (Methodology) ↔ Section IV-D (Proposed Solution) — both describe the architecture (one code-focused, one math-focused). At minimum, ensure no sentences are repeated verbatim.
- [ ] **Compress Implementation Plan** (Section III subsection) — currently lists 5 steps that are repeated in Methodology

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
  - These can be removed from `Project-Attachment-GroupXX.zip` even if kept in repo

---

## 📊 6. FLOPs tables — update with new measurements

After the latest cluster FLOPs run (1111727 or successor) finishes:

- [ ] Pull updated JSONs to local: `scp -P 60441 -r thheim@dnat.simula.no:~/HMR/flops/results flops/`
- [ ] Update **Table V** (Sudoku FLOPs) using `flops/results/*_sudoku.json`
- [ ] Update **Table VI** (Maze FLOPs) using `flops/results/*_maze.json`
- [ ] Regenerate plots: `python3 flops/flops.py plot --results-dir flops/results`
- [ ] Verify both PNGs show only 4 models per task (no SHREK Tiny phantom, no TRM MLP)
- [ ] Replace `sudoku_accuracy_vs_flops.png` and `maze_accuracy_vs_flops.png` references in `main.tex` if any path issues

---

## ⚠️ 7. main.tex — content questions to verify

- [ ] **Ablation Table IV "HRM + EMA" 57%** — chart we verified shows ~67% peak. Either:
  - Re-verify the source of the 57% number (separate run? different metric?)
  - OR add a footnote acknowledging the gap
  - OR replace with the verified ~67% from chart
- [X] **Section II-A end** says HRM compared "exactly comparing apples to apples" issue — review wording, currently a bit informal
- [X] **Section III research gap "Gap 2: Gap 2"** (line 167) — duplicated label, fix
- [ ] **Section IV-A "Approach"** — bullet list mentions "Logical Floor" and "Inference-Time Stuck Alarm" (early plan terms) but the actual implementation in Section IV-D uses "error injection" and "stagnation delta" — terminology drift, unify

---

## 📦 8. Submission package — final assembly

- [ ] Add `huggingface_hub` to `requirements.txt`
- [ ] Add `pyyaml` to `requirements.txt`
- [ ] Verify `requirements.txt` contains all needed: `torch`, `flash-attn`, `omegaconf`, `pydantic`, `hydra-core`, `einops`, `tqdm`, `wandb`, `adam-atan2-pytorch`, `huggingface_hub`, `pyyaml`
- [ ] Update `README.md` to include:
  - Setup instructions (`pip install -r requirements.txt`)
  - Hardware requirement (NVIDIA GPU + CUDA 12.6)
  - How to run `python test.py`
  - Expected output table matching Tables II + III
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

---

## 🟢 9. Things that are GOOD — DO NOT change

Leave these alone, they're working well:

- Introduction's logical flow: LLM limits → Green AI → HRM → TRM → SHREK
- Ablation Table IV structure (component breakdown is clear)
- FLOPs analysis (Tables V/VI + accuracy-vs-FLOPs figures) — strongest paper contribution
- Honest limitations section (acknowledges Maze costs more, no hyperparameter search, etc.)
- Future Work section (concrete directions: ARC-AGI, halting improvements, energy measurements)
- Math notation (flip rate equation, error injection formula, total loss)
- Acknowledgment of dropped Augmented HRM section title

---

## ⏱️ Estimated cleanup impact

| Action | Source lines saved | PDF pages saved |
|---|---|---|
| Delete trailing annotation block | ~60 | 0 (source-only) |
| Drop 3 benchmark subsections | ~25 | ~1.5 |
| Cut Discussion redundancy | ~8 | ~0.3 |
| Delete intro repeat + duplicated paragraphs | ~5 | ~0.2 |
| Spelling + factual fixes | — | reviewer-proof |
| **Total** | **~100 source** | **~2 PDF pages** |

---

## 🎯 Recommended priority order

1. **Must-fix factual errors** (Section 1) — placeholders, wrong citations, count mismatches
2. **Naming consistency** (Section 2) — SHREK Small vs Tiny, TRM MLP decision
3. **Delete trailing annotation block** (Section 3 first item) — cosmetic but easy
4. **Drop 3 benchmark subsections** (Section 3) — biggest readability win
5. **Direct redundancies** (Section 3) — intro repeat, duplicated paragraphs
6. **Cut Discussion** to remove overlap with Conclusion
7. **Spelling pass** (Section 5)
8. **Update FLOPs tables** with latest measurements (Section 6)
9. **Repo cleanup** (Section 4) — empty folders, stale JSONs
10. **README + requirements.txt + final test.py verification** (Section 8)
11. **Build ZIP and submit** (Section 8)
