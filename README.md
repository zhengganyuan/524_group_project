# Mental Health Status Detection from Social Media Text

Multi-class classification of mental health conditions based on social media statements, using both traditional ML (XGBoost) and deep learning (TextCNN) approaches.

## Dataset

**Source:** [Kaggle - Sentiment Analysis for Mental Health](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health)

| Attribute | Detail |
|-----------|--------|
| Raw samples | ~53,043 rows |
| Clean samples | ~51,059 rows (after deduplication and noise removal) |
| Features | `statement` (free-text social media post) |
| Labels | `status` (7 mental health categories) |

**7 Classes:** Normal, Depression, Suicidal, Anxiety, Bipolar, Stress, Personality disorder

## Project Structure

```
524_Group_Project/
├── Combined Data.csv           # Raw dataset
├── Combined Data.xlsx          # Raw dataset
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── src/
│   ├── data_loader.py          # Data loading & cleaning pipeline
│   ├── eda.py                  # Exploratory data analysis (6 plots)
│   ├── preprocessing.py        # Text preprocessing & data splitting
│   ├── feature_engineering.py  # TF-IDF + handcrafted feature extraction
│   ├── train.py                # Unified training entry point
│   ├── evaluate.py             # Evaluation metrics & visualization
│   └── models/
│       ├── xgboost_model.py    # XGBoost classifier
│       └── neural_net.py       # TextCNN (PyTorch)
├── data/
│   └── glove.6B.100d.txt       # GloVe embeddings (download separately)
├── models/                     # Saved trained model weights
└── figures/                    # Generated plots and charts

```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Python version:** 3.10+ recommended

### 2. Download GloVe Embeddings (Optional)

GloVe pre-trained word vectors improve TextCNN performance. If skipped, the model falls back to random initialization.

**Option A — Manual download:**

1. Download from https://nlp.stanford.edu/data/glove.6B.zip (~862 MB)
2. Extract `glove.6B.100d.txt` (~330 MB)
3. Place it at `data/glove.6B.100d.txt`

**Option B — Command line:**

```bash
mkdir -p data
curl -Lo data/glove.6B.zip https://nlp.stanford.edu/data/glove.6B.zip
unzip data/glove.6B.zip glove.6B.100d.txt -d data/
rm data/glove.6B.zip
```

**Option C — PowerShell (Windows):**

```powershell
New-Item -ItemType Directory -Force -Path data
Invoke-WebRequest -Uri "https://nlp.stanford.edu/data/glove.6B.zip" -OutFile "data\glove.6B.zip"
Expand-Archive -Path "data\glove.6B.zip" -DestinationPath "data" -Force
Remove-Item "data\glove.6B.zip"
# Only keep the 100d version
Remove-Item "data\glove.6B.50d.txt", "data\glove.6B.200d.txt", "data\glove.6B.300d.txt" -ErrorAction SilentlyContinue
```

### 3. Prepare Dataset

Place `Combined Data.csv` in the project root directory (already included).

## How to Run

Execute the three scripts **in order** from the project root:

```bash
# Step 1: Exploratory Data Analysis — generates 6 plots to figures/
python src/eda.py

# Step 2: Train all models — Baselines (Majority Class + Naive Bayes) + XGBoost + TextCNN
python src/train.py

# Step 3: Evaluate & visualize — generates comparison table + evaluation plots
python src/evaluate.py
```

### Expected Output

After running all three scripts:

- `figures/` — 12 plots total (6 EDA + 6 evaluation)
- `models/` — Saved model weights (`.pkl` + `.pt`)
- Console — Performance comparison table with Accuracy, Macro-F1, Weighted-F1, Macro-AUC

## Models

| Model | Type | Description |
|-------|------|-------------|
| Majority Class | Baseline | Always predicts the most frequent class |
| Naive Bayes | Baseline | MultinomialNB on TF-IDF features |
| XGBoost | Traditional ML | Gradient boosting on TF-IDF + handcrafted features, with SMOTE |
| TextCNN | Neural Network | Multi-kernel CNN (PyTorch) with GloVe embeddings |

## Evaluation Metrics

- Accuracy
- Precision / Recall / F1-score (per-class + macro/weighted)
- AUC-ROC (One-vs-Rest, multi-class)
- Confusion Matrix (7x7 heatmap)

## Key Design Decisions

- **No stemming** — Full word forms preserve semantic meaning critical for mental health text
- **SMOTE applied only to training set** — Prevents data leakage into test set
- **Stratified split (80/20)** — Maintains class proportions across train/test
- **Class-weighted loss** — Both XGBoost and TextCNN use balanced class weights
- **GloVe 6B 100d** — Lightweight pre-trained embeddings suitable for informal social media text

## Technical Stack

- Python 3.10+
- pandas, numpy, scikit-learn
- XGBoost
- PyTorch (TextCNN)
- matplotlib, seaborn
- imbalanced-learn (SMOTE)
- NLTK (lemmatization)
