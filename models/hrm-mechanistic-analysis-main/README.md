# Are Your Reasoning Models Reasoning or Guessing? A Mechanistic Analysis of HRM

<img width="993" height="359" alt="image" src="https://github.com/user-attachments/assets/2425608d-f56f-4c7a-b426-f7e1227202dc" />

Mechanistic analysis of **Hierarchical Reasoning Models (HRM)** on the *Sudoku-Extreme* dataset.  
This repository provides the **official PyTorch implementation** of our 2026 paper:

> **"Are Your Reasoning Models Reasoning or Guessing? A Mechanistic Analysis of Hierarchical Reasoning Models"**  
> *Zirui Ren, Ziming Liu*  
> arXiv:2601.10679 | [PDF](https://arxiv.org/abs/2601.10679)

## Quick Start 🚀

### Prerequisites ⚙️

We use Git LFS to store the example checkpoints; so if you want to use them, install via
~~~
git lfs install
~~~
and **clone this repository with the variable set as below**, so that you don't download all large files all at once.
~~~
export GIT_LFS_SKIP_SMUDGE=1
git clone git@github.com:renrua52/hrm-mechanistic-analysis.git
~~~

Make sure you have PyTorch, CUDA and FlashAttention installed (see the official guide of [HRM](https://github.com/sapientinc/HRM)). Then install the python dependencies:
~~~
pip install -r requirements.txt
~~~

### Dataset Preparation 📊

Run the following commands to build vanilla and augmented sudoku dataset, respectively.

~~~
python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000 --subsample-size 1000 --num-aug 1000
python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000-hint --subsample-size 1000 --num-aug 1000 --hint
~~~

The latter is an augmented version of the former, with easier sudoku puzzles mixed in. In our paper, we showed that this augmentation helps to restore inference stability.

### Example Checkpoints 🚧

For those not interested in training models themselves, we provide two sets of trained checkpoints for evaluation. Run the following commands to download them (**you only need the first one to run the demo**):
~~~
# 1 single vanilla HRM ckpt
git lfs pull --include="checkpoint_example/Sudoku-extreme-1k-aug-1000 ACT-torch/**"
# 10 ckpts of HRM trained on augmented dataset. This is for the model boostrapping technique.
git lfs pull --include="checkpoint_example/Sudoku-extreme-1k-aug-1000-hint ACT-torch/**"  
~~~

### Quick Demo of Reasoning Trajectory & Visualization 🎨

Check out our [demo notebook](./demo.ipynb) to understand how most results in the paper were attained!

We added a `require_trace` argument to the HRM model forwarding process, with the intermediate z_H states returned as a list. The `visualization` module is used to visualize both reasoning trajectory and error landscape. 

## Model Evaluation 📈

Most of the utilities for testing are implemented in `eval_utils.py`.

For evaluating trained checkpoints, we provide a quick-and-dirty implementation of batched inference in `batch_inference.py`, which runs on a single GPU. It supports the evaluation on *Sudoku-Extreme* of ensembled checkpoints, with or without the permuting token augmentation. For example, to do a full evaluation of the example checkpoint, run:
~~~
python batch_inference.py \
--checkpoints "checkpoint_example/Sudoku-extreme-1k-aug-1000 ACT-torch/HierarchicalReasoningModel_ACTV1 pastel-lorikeet/example_checkpoint"
~~~
which typically takes 30 mins on an A10 GPU to reproduces the ~55% accuracy in the original HRM paper.

Multiple forwarding requires way more time. To get a taste of the accuracies of Table 1 in the paper, we recommend running the single-GPU script on a small portion of test samples. You can do something like:
~~~
python batch_inference.py \
--checkpoints "checkpoint_example/Sudoku-extreme-1k-aug-1000 ACT-torch/HierarchicalReasoningModel_ACTV1 pastel-lorikeet/example_checkpoint" \
--permutes 9 --num_batch 10 --batch_size 100
~~~
which tests the designated ckpt on 1000 test samples, applying 9 token permutations to each.

For the full result of **Augmented HRM**, do one of the following:
- Download the second set of example checkpoints.
- Train your own series of checkpoints on the *augmented* dataset with ckpt interval 1000. Replace the checkpoints in `scripts/example_evaluation.sh` with your own.

Then run the script
~~~
bash ./scripts/example_evaluation.sh
~~~
This is a snapshot evaluation for you to understand how augmentation works. The full evaluation (see the [script](/scripts/example_evaluation.sh)) process takes around 500 GPU hours, due to the cost of multiple forward processes. Parallelization should speed up the evaluation significantly - contributions are welcome!

Due to large variances in small-sample training, a ~2% discrepancy in single ckpt results and ~4% in multiple ckpt results are considered acceptable.

## Training ⚡️

Log in [Weights & Biases](https://wandb.ai) via
~~~
wandb login
~~~

Run the following commands to train HRM on both version of the datasets.

~~~
OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 pretrain.py data_path=data/sudoku-extreme-1k-aug-1000 epochs=40000 eval_interval=1000 lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0
OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 pretrain.py data_path=data/sudoku-extreme-1k-aug-1000-hint epochs=40000 eval_interval=1000 lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0
~~~

Training randomness has observable impact on the outcome, so we recommend inspecting the `exact_accuracy` in W&B and choosing the best checkpoint for evaluation.

The `eval_interval` option does not influence training process. Evaluation typically takes considerable time, so set it wisely. If you wish to ensemble ckpts, however, we recommend checkpointing more frequently in the final stage of training.

## Acknowledgement 🧑‍🎓

This repository was forked-and-hacked from [sapientai/HRM](https://github.com/sapientinc/HRM) (Apache-2.0).
I kept most upstream code intact and mostly layered on evaluation, augmentation and visualisation.  

Core files I worked on include:
- `eval_utils.py`
- `batch_inference.py`  
- `visualization/`
- `demo.ipynb`

Besides, I slightly modified the dataset building code for augmentation, and modeling code for reasoning trace extraction.

Most original logic remains unchanged; see the upstream repo for the core implementation.  

## Citation 📜

~~~
@misc{ren2026reasoningmodelsreasoningguessing,
      title={Are Your Reasoning Models Reasoning or Guessing? A Mechanistic Analysis of Hierarchical Reasoning Models}, 
      author={Zirui Ren and Ziming Liu},
      year={2026},
      eprint={2601.10679},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2601.10679}, 
}
~~~
