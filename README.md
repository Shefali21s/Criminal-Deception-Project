# Criminal Sentiment & Deception Detection — Feature Engineering

## Overview

This project extracts psychological and emotional cues from text data to support deception detection in a criminal investigation context. Using NLP techniques and pre-built lexicons, sentiment and emotion features are engineered from a combined, labeled deception dataset to improve classification accuracy in distinguishing deceptive from truthful statements.

---

## Dataset

| Field | Detail |
|---|---|
| **File** | `combined_clean_preprocessed.csv` |
| **Sources** | Deceptive Opinion Spam dataset, MU3D, LIAR |
| **Target label** | `deceptive` / `truthful` |
| **Text column** | `clean_text` (pre-tokenized, cleaned, and normalized) |

---

## Feature Engineering

### Sentiment Features — VADER

Four scores extracted per statement using `nltk`'s `SentimentIntensityAnalyzer`:

| Feature | Description | Type |
|---|---|---|
| `neg` | Negative polarity score | Float (0–1) |
| `neu` | Neutral polarity score | Float (0–1) |
| `pos` | Positive polarity score | Float (0–1) |
| `compound` | Composite sentiment score | Float (−1–1) |

```python
from nltk.sentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()
df[['neg', 'neu', 'pos', 'compound']] = df['clean_text'].apply(
    lambda x: pd.Series(sia.polarity_scores(x))
)
```

### Emotion Features — NRC Lexicon

Eight binary/count indicators extracted per statement using the NRC Emotion Lexicon:

`anger`, `fear`, `joy`, `trust`, `anticipation`, `disgust`, `surprise`, `sadness`

```python
nrc = pd.read_csv("NRC-Emotion-Lexicon.txt", sep='\t', names=["word", "emotion", "association"])
nrc = nrc[nrc["association"] == 1]
# Tokenize each sentence and count matching words per emotion category
```

---

## Setup

### Prerequisites

- Python 3.x
- Libraries: `pandas`, `nltk`

### Installation

```bash
pip install pandas nltk
```

```python
import nltk
nltk.download('vader_lexicon')
```

### Required Files

- `combined_clean_preprocessed.csv` — the preprocessed input dataset
- `NRC-Emotion-Lexicon.txt` — downloadable from [saifmohammad.com](http://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm)

---

## Output

The pipeline produces `combined_features.csv`, which extends the original dataset with all engineered sentiment and emotion columns, ready for downstream modeling.

---

## Motivation

Psychological research suggests deceptive text tends to exhibit elevated fear/anger, lower positive sentiment, and reduced emotional trust and joy. The engineered features are designed to surface these patterns for ML classifiers.

---

## Next Steps

- Visualize feature distributions across labels
- Correlate features with the target label
- Train ML models (e.g., Random Forest, Gradient Boosting) using the engineered features

---

## References

- Hutto, C. & Gilbert, E. (2014). *VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text.*
- Mohammad, S. M., & Turney, P. D. (2013). *Crowdsourcing a Word–Emotion Association Lexicon.*
- [NLTK Documentation](https://www.nltk.org)
- [NRC Emotion Lexicon](http://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm)
