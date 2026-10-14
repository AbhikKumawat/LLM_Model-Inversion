# Language Model Inversion Using Traditional Machine Learning

## Overview

This project explores whether next-token probability distributions from a pretrained language model can help identify or approximate the original text prompt using traditional machine learning techniques.

Inspired by the research paper [Language Model Inversion](https://arxiv.org/abs/2311.13647), the project investigates an ML-based approximation using Python, Scikit-learn, TF-IDF, regression algorithms, and cosine similarity.

The primary objective is to understand how probability-derived features can be used to retrieve similar prompts from a known collection.

**Important:** This project is an educational approximation of the research paper, not a reproduction of its original deep-learning-based inversion method.

## Objectives

- Extract numerical features from next-token probability distributions.
- Apply traditional machine learning to predict text representations.
- Compare Ridge Regression and Random Forest Regression.
- Retrieve candidate prompts using cosine similarity.
- Evaluate predictions using exact-match accuracy and token-level metrics.
- Study how dataset size, temperature, and top-k filtering affect retrieval.
- Understand the limitations of reconstructing text from probability information.

## Technologies Used

- Python
- Scikit-learn
- NumPy and Pandas
- Matplotlib
- Hugging Face Transformers
- PyTorch, used only to run the frozen pretrained language model as a probability source

The inversion and retrieval components use traditional machine learning. No neural network is trained for prompt reconstruction.

## Project Workflow

1. **Dataset creation:** Generate a synthetic dataset of authorized prompts.
2. **Probability extraction:** Use a frozen pretrained language model to obtain next-token probability distributions.
3. **Feature engineering:** Extract probability vectors and statistical features, including entropy and maximum probability.
4. **Dimensionality reduction:** Apply Truncated SVD to reduce the dimensionality of probability features.
5. **Traditional ML modeling:** Train Ridge Regression and Random Forest Regression to predict TF-IDF text representations.
6. **Candidate retrieval:** Use cosine similarity and nearest-neighbor search to retrieve the closest prompt from the training collection.
7. **Evaluation:** Compare retrieved candidates with the original held-out prompts.
8. **Visualization:** Generate a chart comparing the evaluation metrics.

## Algorithms

### 1. Ridge Regression

Ridge Regression predicts a TF-IDF representation from probability-derived features while applying L2 regularization to reduce overfitting.

### 2. Random Forest Regression

Random Forest Regression provides a second classical ML approach that can model nonlinear relationships between features and text representations.

### 3. TF-IDF and Cosine Similarity

TF-IDF represents prompts numerically using word and phrase frequencies. Cosine similarity measures the similarity between the predicted representation and candidate prompt representations.

## Project Structure

```text
traditional_ml_language_model_inversion/
├── requirements.txt
├── config.py
├── dataset.py
├── features.py
├── model.py
├── train.py
├── evaluate.py
├── predict.py
├── visualize.py
├── README.md
└── report.md
```

## Installation

Python 3.10 or later is recommended.

Clone the repository or download and extract the project ZIP.

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment.

Windows:

```bash
.venv\Scripts\activate
```

macOS or Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The first run downloads the pretrained model and tokenizer. An internet connection is required unless the model is already cached locally.

## Running the Project

### Step 1: Inspect the dataset

```bash
python dataset.py
```

### Step 2: Train and evaluate the models

```bash
python train.py
```

The script extracts probability features, creates reproducible training and testing splits, trains both regression approaches, and saves evaluation results.

### Step 3: Generate visualizations

```bash
python visualize.py
```

The chart is saved to:

```text
figures/model_comparison.png
```

### Step 4: Demonstrate prompt retrieval

```bash
python predict.py
```

This demonstrates retrieval of a candidate prompt for a held-out example.

## Experiments

The project supports experiments with dataset size, temperature, and top-k filtering.

Dataset size:

```bash
python train.py --limit 40
python train.py --limit 80
python train.py --limit 100
```

Temperature:

```bash
python train.py --temperature 0.7
python train.py --temperature 1.0
python train.py --temperature 1.5
```

Top-k filtering:

```bash
python train.py --top-k 5
python train.py --top-k 20
python train.py --top-k 50
```

The default random seed is fixed for reproducibility. Save the output metrics after each experiment because subsequent runs overwrite the same artifact files.

## Evaluation Metrics

- **Exact-match accuracy:** Measures the fraction of retrieved prompts identical to the original test prompts.
- **Token precision:** Measures the proportion of predicted tokens that overlap with the reference.
- **Token recall:** Measures the proportion of reference tokens recovered by the prediction.
- **Token F1-score:** Combines token precision and recall.
- **Retrieval cosine similarity:** Measures the similarity between the predicted text representation and the retrieved candidate.

These metrics measure different aspects of performance. High cosine similarity does not necessarily mean the original prompt was reconstructed correctly.

## Example Observation

In one experiment, the original prompt was:

> Give a beginner-friendly guide to an electric bicycle.

The retrieved candidate was:

> Give a beginner-friendly guide to a city park.

The measured cosine similarity was approximately 0.6907.

The shared wording produced a relatively high similarity, but the subject was different. This illustrates why similarity-based retrieval should not be confused with exact prompt reconstruction.

## Scientific Limitations

1. A next-token probability distribution does not uniquely identify an entire prompt in general.
2. Different prompts can produce similar probability distributions.
3. The current system retrieves candidates only from the training prompt collection.
4. Unique held-out prompts cannot be retrieved exactly when they are absent from the candidate bank.
5. Token overlap and cosine similarity do not establish semantic equivalence.
6. The synthetic dataset is small and does not establish generalization to arbitrary real-world prompts.
7. The project does not reproduce the original paper's deep-learning-based inversion method.

The original test prompt is retained as ground truth for evaluation. The prediction stage receives probability-derived features rather than the original prompt string.

## Future Improvements

- Evaluate larger and more diverse authorized prompt datasets.
- Compare against nearest-neighbor retrieval directly on probability features.
- Test additional random seeds and report variability.
- Investigate alternative dimensionality-reduction methods.
- Compare closed-bank retrieval with more realistic held-out-template experiments.
- Improve probability-feature engineering and retrieval evaluation.
- Study how access to different probability information affects prompt identification.

## Research Reference

Paper: [Language Model Inversion](https://arxiv.org/abs/2311.13647)


