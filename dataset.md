On the cluster:

# ARC-AGI-1

git clone https://github.com/fchollet/ARC-AGI ~/arc-agi-1
cp -r ~/arc-agi-1/data ~/HMR/models/OurMODEL/dataset/raw-data/ARC-AGI/

# ConceptARC

git clone https://github.com/victorvikram/ConceptARC ~/conceptarc
cp -r ~/conceptarc/corpus ~/HMR/models/OurMODEL/dataset/raw-data/ConceptARC/

# ARC-AGI-2

git clone https://github.com/arcprize/ARC-AGI-2 ~/arc-agi-2
cp -r ~/arc-agi-2/data ~/HMR/models/OurMODEL/dataset/raw-data/ARC-AGI-2/

# Then build

cd ~/HMR/models/OurMODEL/dataset/
python3 build_arc_dataset.py

So the dataset situation is:

- Sudoku → HuggingFace, fully automatic
- Maze → HuggingFace, fully automatic
- ARC-AGI-1/2 → manual git clone, then build script
