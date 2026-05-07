# README

------------------------------------------------------------

## 1. Project Overview

Project Title: Mohamed qqq models.

Model Type:
Transformer with sentiment fusion, trained in four generations, namely a 113,219-parameter price-only baseline, my first model v1 (323,107 parameters with a Long Short-Term Memory branch on the price block), my second model v2 (360,198 parameters with a parallel sigmoid direction head), and my third model v3 (369,106 parameters with a MANA-Net attention pool over per-headline FinBERT vectors plus macro and technical features).

Objective:
Regression of forward percent returns of the Invesco QQQ Trust Exchange-Traded Fund at the one-, three- and five-day horizons, with auxiliary classification heads in my second and third models.

Dataset Used:
QQQ daily price data from 2000 to 2024 sourced from Yahoo Finance, plus a 45,561-headline financial news corpus dominated by the New York Times Archive at approximately ninety-eight percent of the entries, both bundled in the `data/` folder of this submission.

Expected test evaluation for sanity check: Cohort backtest final at t+3 from $10,000 of starting capital of $16,999 for my first model v1, against buy-and-hold's $15,842.

------------------------------------------------------------

## 2. Repository Structure

```
Mohamed qqq models/
  Mohamed qqq models.py    The single-file pipeline that builds the FinBERT cache, trains the four models, and runs the cohort backtest end to end.
  README.md                This file.
  data/                    The two source CSV files, namely the QQQ daily price file and the 45,561-headline corpus.
  src/                     My importable modules, namely the data loaders, the four model definitions, the three training loops, and the evaluation utilities that the pipeline depends on.
  notebooks/               Six numbered notebooks plus a helpers module, namely the modular Jupyter version of the pipeline.
  results/                 The prebuilt artifacts from my run, namely the FinBERT cache, the four trained checkpoints, every result CSV, and every figure produced by the pipeline.
```

------------------------------------------------------------

## 3. Dataset

### OPTION A. PUBLIC DATASET SPLITS

Dataset Link:
The two source CSV files are bundled in the `data/` folder of this submission, originally compiled from Yahoo Finance for the QQQ price data and the New York Times Archive API for the headline corpus.

Where to place the downloaded dataset:
```
data/
  QQQ_2000_2024_with_MACD.csv
  financial_headlines_2000_2024.csv
```

------------------------------------------------------------

## 4. Model Checkpoint

Box Link to Best Model Checkpoint:
The four trained checkpoints are bundled in `results/models/` of this submission, so no Box download is required to reproduce the backtest.

Give access to:
yusun@usf.edu, kandiyana@usf.edu

Where to place the checkpoint after downloading:
```
results/models/
  baseline.pt          (price-only baseline)
  mohamed.pt           (my first model v1)
  mohamed_v2.pt        (my second model v2)
  mohamed_v3.pt        (my third model v3)
```

------------------------------------------------------------

## 5. Requirements (Dependencies)

Python Version:
3.10 or higher.

How to install all dependencies:
The dependencies are numpy, pandas, matplotlib, torch, scikit-learn, scipy, transformers, yfinance and pyarrow.

Using pip:
```
pip install numpy pandas matplotlib torch scikit-learn scipy transformers yfinance pyarrow
```

Using conda (creates env and installs):
```
conda create -n mohamed-qqq python=3.10
conda activate mohamed-qqq
pip install numpy pandas matplotlib torch scikit-learn scipy transformers yfinance pyarrow
```

------------------------------------------------------------

## 6. Running the Test Script

Run the cohort backtest from $10,000 of starting capital with five basis points of round-trip transaction cost, which loads the four bundled checkpoints, generates per-horizon predictions on the held-out test window, and writes the summary CSV plus the per-horizon equity plots to `results/`.

```
python "Mohamed qqq models.py" --backtest
```

------------------------------------------------------------

## 7. Running the Training Script

Build the FinBERT and macro caches once, train the four models in sequence, and then run the cohort backtest from $10,000 of starting capital, all in a single command.

```
python "Mohamed qqq models.py" --all
```

Optional arguments (if supported):
- `--build-caches`: build only the FinBERT and macro caches.
- `--train {baseline,v1,v2,v3,all}`: train one model or all four.
- `--force-cache`: rebuild the caches even if they exist.
- `--backtest`: run only the cohort backtest from $10,000.

------------------------------------------------------------

## 8. Submission Checklist

- [x] Dataset provided using Option A and bundled in the `data/` folder.
- [x] Model checkpoint bundled in `results/models/` (Box link not required).
- [x] Python 3.10 or higher specified, dependencies listed in section 5.
- [x] Test command works (`python "Mohamed qqq models.py" --backtest`).
- [x] Train command works (`python "Mohamed qqq models.py" --all`).

------------------------------------------------------------
