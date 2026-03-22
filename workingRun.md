cd ~/HMR/models/HRM\(Original\)/HRM-main && DISABLE_COMPILE=1 OMP_NUM_THREADS=8 python3 pretrain.py data_path=../../../dataset/data/sudoku-extreme-1k-aug-1000 epochs=20000 eval_interval=2000 global_batch_size=384 lr=7e-5 puzzle_emb_lr=7e-5
weight_decay=1.0 puzzle_emb_weight_decay=1.0
