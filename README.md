# Unsupervised Clinical Language Translation

- Edited by Wei-Hung Weng (MIT CSAIL)
- Created: Jan 16, 2019
- Latest update: May 10, 2019
- Please contact the author with errors found.
- `ckbjimmy {AT} mit {DOT} edu`

This repository contains the codes for the clinical language translation task using unsupervised SMT method presented in the paper [Unsupervised Clinical Language Translation](https://arxiv.org/abs/1902.01177) (KDD 2019).

### Dependency
- Python 3
- NumPy
- PyTorch
- [Moses](http://www.statmt.org/moses/?n=moses.releases) (for text tokenization, statistical machine translation and back-translation---we suggest to use binary version to prevent complicated compiling process, the version use in the current code is for Ubuntu 17.04+)
- [fastText](https://github.com/facebookresearch/fastText) (for learning word embeddings)
- [MUSE](https://github.com/facebookresearch/muse) (for learning bilingual dictionary induction)

### Quick Start
1. Prepare your corpora in source (`pro`) and target (`con`) languages
2. Edit the data path and parameters in the first section in `run.sh`
3. Replace the line 59-63 of `MUSE/src/evaluation/word_translation.py` with `word1, word2 = line.rstrip().split()` to ensure the correct evaluation
4. Run `run.sh`

The result will be saved in the folder `res/RESULT_DIR` under the root directory. The translated sentence will be saved in the last forward translation folder and named as `test.tgt.hyp.tok`.

### Arguments
```
DATA_PATH           # corpus path (please name the source corpus named as `pro` and target corpus as `con`)
EMB_DIM             # dimension of the word embeddings (default = 300)
SUBWORD             # whether to use subword information when training monolingual word embeddings (default = false)
USE_NEWSCRAWL       # whether to use third-party large-scale corpora for training consumer LM (default = false)
N_THREADS           # number of threads in data preprocessing (default = 48)
IDENTICAL_CHAR      # whether to use identical chars as anchors when training cross-lingual embeddings (default = true)
RESULT_DIR          # where the experimental results will be stored
SENT_BT             # monolingual data used in back-translation (default = 100000)
```

If you use this code, please kindly cite the paper for this GitHub project (see below for BibTex):

```
@inproceedings{weng2019unsupervised,
  title={Unsupervised Clinical Language Translation},
  author={Weng, Wei-Hung and Chung, Yu-An and Szolovits, Peter},
  booktitle={KDD},
  year={2019}
}
```
