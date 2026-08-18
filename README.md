# MusicX — Music habits and self-reported mental health

[![Python](https://img.shields.io/badge/python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![pandas](https://img.shields.io/badge/pandas-EDA-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-baselines-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

<p align="center">
  <img src="assets/banner.png" alt="MusicX Mental Health" width="720">
</p>

Survey analysis of **736** listeners: how often they play 16 genres, and how they rate anxiety, depression, insomnia, and OCD on a 0–10 scale.

The interesting result is not a high R² — it is that **listening habits barely predict symptom scores**, while **three in four people still say music helps**.

---

## Findings (the actual numbers)

| Claim | Evidence in this sample |
| --- | --- |
| People *believe* music helps | **542 / 728** (74.5%) report that music **improves** their mental health. 169 say no effect. **17** say it worsens it. |
| Anxiety is the loudest symptom | Means: Anxiety **5.84**, Depression **4.80**, Insomnia **3.74**, OCD **2.64**. Anxiety and depression move together (`r = 0.52`). |
| Genre ↔ symptoms is weak | Strongest Pearson links: Rock–Depression **0.19**, Metal–Depression **0.18**, Metal–Insomnia **0.16**. Everything else is smaller. |
| The “worsen” group is small and darker | Those 17 people average Depression **7.2** vs **4.9** among people who say music improves things. Flag, not a prevalence rate. |
| The sample is not “the public” | Median age **21**, **62%** Spotify, survey spread on forums in late 2022. |

A random forest trained on 16 genre frequencies + age + hours **does not beat** predicting the training-set mean for anxiety, depression, or insomnia (test R² is slightly *worse* than a dummy regressor). Weak signal is the finding.

---

## What was wrong with the leaked target

The original target `TotalMusicFreq` was the **sum of four genre-frequency columns**. Those same columns were then used as features. Linear regression recovered coefficients `[1, 1, 1, 1]` and an MSE of ~`10⁻²⁹`. That is data leakage: the model was adding four numbers it had already been given.

This analysis instead:

1. Stops filling missing Age / BPM / Music effects with `0` (that invented a 0-year-old and a fake “Never”).
2. Treats `BPM = 999,999,999` as an invalid tempo, not a real value.
3. Encodes all **16** genres, not four.
4. Compares every model to a **dummy baseline**.

Full walkthrough: [`analysis.ipynb`](analysis.ipynb).

---

## Repository layout

```
app.py                 Streamlit explorer (overview, listeners, associations, models)
analysis.ipynb         Narrative analysis — quality → EDA → leakage demo → honest models
src/data.py            Load, clean, ordinal-encode frequencies
data/mxmh_survey.csv   MxMH public survey (see data/README.md)
tests/test_data.py     Cleaning contracts (no Age=0, BPM outlier, encoding)
```

---

## Run it

Clone the repo (or `cd` into the folder that already contains `app.py`). Python 3.10+.

```bash
git clone https://github.com/PedroCarneiroMarques/MusicX_Emotion_Analysis.git
cd MusicX_Emotion_Analysis

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py               # dashboard
pytest                             # cleaning tests
```

If `streamlit` says `File does not exist: app.py`, you are in the wrong directory. `ls` should show `app.py`, `analysis.ipynb`, and `data/`.

The dashboard does not need an API key. Everything runs on the CSV in `data/`.

---

## Data

[Music & Mental Health Survey Results](https://www.kaggle.com/datasets/catherinerasgaitis/mxmh-survey-results) (MxMH), collected by Catherine Rasgaitis, 27 August – 9 November 2022.

Respondents ranked 16 genres as Never / Rarely / Sometimes / Very frequently, and rated Anxiety, Depression, Insomnia, and OCD from 0 (“I do not experience this”) to 10 (“I experience this regularly, constantly, or to an extreme”).

This repo does not own the survey. Cite the Kaggle dataset if you reuse the CSV.

---

## Limits (read these before quoting a chart)

- **Self-report**, not a clinical diagnosis.
- **Convenience sample** (online form, music communities) — young, streaming-native.
- **Correlation ≠ causation.** A depressed listener may pick metal; metal does not “cause” the score.
- **Tiny “Worsen” cell (n = 17).** Do not percentage that group.
- **BPM** is mostly unused: 107 missing, plus 7 values outside 20–300 BPM (including `999,999,999`).

---

## Stack

pandas · seaborn / matplotlib · Plotly · scikit-learn · Streamlit

---

## Author

**Pedro Carneiro Marques** — Data Analyst (Application Support Analyst, data focus, Airbus).  
[LinkedIn](https://www.linkedin.com/in/slvklv/) · [GitHub](https://github.com/PedroCarneiroMarques)
