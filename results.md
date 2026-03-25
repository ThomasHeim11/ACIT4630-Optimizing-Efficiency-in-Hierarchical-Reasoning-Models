# Results

#### Sudoku Extreme (1k)

### Standard HRM

Results: 0.53 % accuracy,that's within the paper's 55% (±2%) target
![alt text](image-1.png)

### Augmented HRM

![alt text](image-5.png)
![alt text](image-4.png)

### Shrek HRM

![alt text](image-2.png)
![alt text](image-3.png)

#### General results

SHREK Large gets 70.6% without any tricks — the Augmented HRM needs 10 checkpoints + 9 permutations just to
Reach 89.3%. This means:

- SHREK is a better standalone model — deploy one checkpoint, get 70.6%
- The ensemble gives diminishing returns for SHREK because it's already good
- The error injection reduces reliance on expensive inference-time techniques

That's a strong thesis argument: SHREK makes the model more self-sufficient, reducing the need for costly
ensemble methods.

#### Maze Hard (1k)

#### ARC-AGI1 (1k)

#### ARC-AGI2 (1k)
