# Final-Project_NLP

# Mental Health Condition Classification
**Advanced NLP Final Project** (Barcelona School of Economics)
**Instructor:** Arnault Gombert
**Authors:** [your names]

## Objective
Our goal is to classify text into 7 mental health categories: Anxiety, Normal, Depression, Stress, Personality disorder, Bipolar, and Suicidal. 

We focused on how to build effective models with very little data. We limited our training set to just 32 labeled examples, tested techniques to maximize performance, compared this against a model trained on the full dataset, and compressed our best model.

## Dataset
We used the [Mental Health Condition Classification](https://huggingface.co/datasets/sai1908/Mental_Health_Condition_Classification) dataset from HuggingFace. After removing duplicates, we kept about 100,000 rows. We split the data into training (64%), validation (16%), and testing (20%) sets. Every experiment uses the exact same test set for a fair comparison.

## Results (Test Set)

| Stage | Model | Accuracy | Suicidal Recall |
|---|---|---|---|
| 1 | Basic word-matching baseline | 74.5% | 0.45 |
| 2 | BERT (32 labels + distillation) | 75.7% | 0.745 |
| 3 | BERT (Full data) | 91.7% | 0.78 |

*Note: The published state-of-the-art accuracy on this data is 94.83%. We track "Suicidal recall" closely because missing these cases is the most critical error in a real-world screening tool.*

## Repository Structure
* `notebooks/`: Numbered steps from evaluation (00) to final models (04).
* `src/`: Shared code for metrics, processing, and plotting.
* `config/`: Model settings and hyperparameters.
* `data/`: Ready-to-use dataset splits.
* `results/`: Saved metrics, predictions, and figures.
* `tests/`: Checks to ensure data remains consistent.

## How to Run It

1. Set up your environment and install the required packages:
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
