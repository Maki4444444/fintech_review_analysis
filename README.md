# Fintech Review Analytics

A data engineering and NLP pipeline that scrapes, analyzes, and visualizes Google Play Store reviews for three Ethiopian banks — Commercial Bank of Ethiopia (CBE), Bank of Abyssinia (BOA), and Dashen Bank — to generate actionable customer experience insights.

Built as part of the **10 Academy AI Mastery Program — Week 2 Challenge**.

---

## Project Structure

fintech_review_analysis/
├── .vscode/
│   └── settings.json
├── .github/
│   └── workflows/
│       └── unittests.yml
├── .gitignore
├── requirements.txt
├── README.md
├── data/
│   ├── raw/
│   └── clean/
├── notebooks/
│   ├── init.py
│   ├── Scraping.ipynb
│   ├── preprocessing.ipynb
│   ├── sentiment.ipynb
│   └── thematic.ipynb
├── scripts/
│   ├── init.py
│   ├── scrape.py
│   ├── preprocess.py
│   ├── sentiment.py
│   └── thematic.py
├── src/
│   └── init.py
└── tests/
└── init.py

---

## Setup

### Prerequisites
- Python 3.11+
- Git

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/<your-username>/fintech_review_analysis.git
cd fintech_review_analysis
```

**2. Create and activate virtual environment**
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

---

## Tasks

### Task 1: Data Collection & Preprocessing

#### Scraping Methodology

Reviews were collected from the Google Play Store using the [`google-play-scraper`](https://pypi.org/project/google-play-scraper/) Python library. The following three Ethiopian bank apps were targeted:

| Bank | App ID |
|------|--------|
| Commercial Bank of Ethiopia | `com.combanketh.mobilebanking` |
| Bank of Abyssinia | `com.boa.boaMobileBanking` |
| Dashen Bank | `com.dashen.dashensuperapp` |

Each app was scraped with the following parameters:
- `lang='en'` — English language reviews
- `country='et'` — Ethiopian Play Store
- `sort=Sort.NEWEST` — most recent reviews first
- `count=600` — 600 reviews requested per bank to ensure the 400-review minimum is met after cleaning

#### Date Range

Reviews were scraped in **May 2026**. The date range of collected reviews depends on availability from the Play Store and varies per bank. The exact earliest and latest dates per bank are documented in `notebooks/02_preprocessing.ipynb` under the post-cleaning quality check section.

#### Preprocessing Steps

1. **Amharic Translation** — Reviews containing Ethiopic Unicode characters (`\u1200–\u137F`) were detected and translated to English using `deep-translator`'s Google Translate backend. Emojis were preserved as sentiment indicators.
2. **Deduplication** — Duplicate reviews were removed by review ID.
3. **Null Removal** — Rows missing review text or rating were dropped.
4. **Date Normalization** — All dates normalized to `YYYY-MM-DD` format.
5. **Column Selection** — Final dataset contains five columns: `review`, `rating`, `date`, `bank`, `source`.

#### Output

| File | Description |
|------|-------------|
| `data/raw/reviews_raw.csv` | Raw scraped reviews (1,800 total) |
| `data/clean/reviews_clean.csv` | Combined cleaned dataset (1,800 reviews) |
| `data/clean/commercial_bank_of_ethiopia_clean.csv` | CBE reviews (600) |
| `data/clean/bank_of_abyssinia_clean.csv` | BOA reviews (600) |
| `data/clean/dashen_bank_clean.csv` | Dashen reviews (600) |

> Note: All CSV files are listed in `.gitignore` and are not committed to GitHub.

#### Limitations

- `langdetect` was initially used for Amharic detection but proved unreliable, misclassifying Amharic text as Somali, Afrikaans, Polish, and other languages. It was replaced with direct Ethiopic Unicode range detection (`\u1200–\u137F`).
- `google-play-scraper` returns English-language reviews by default (`lang='en'`). Reviews written in Amharic by users still appeared in the dataset, suggesting the Play Store does not strictly filter by language.
- Review availability is limited by what the Play Store exposes via the scraper — older reviews may not be accessible.
- 78 Amharic reviews were detected and translated across all three banks.

---

### Task 2: Sentiment & Thematic Analysis

#### Sentiment Analysis

Two models were used and compared:

| Model | Type | Description |
|-------|------|-------------|
| VADER | Lexicon-based | Fast, rule-based, good for short informal text |
| DistilBERT | Transformer | Context-aware, higher accuracy, fine-tuned on SST-2 |

**Label thresholds (VADER):**
- `compound >= 0.05` → positive
- `compound <= -0.05` → negative
- otherwise → neutral

**Model Agreement Rate:** 68.89% overall (CBE: 72.5%, BOA: 62.5%, Dashen: 71.7%)

#### Thematic Analysis

Themes were identified per bank using TF-IDF and spaCy keyword extraction. Each bank has 7 distinct themes supported by keyword examples. Analysis was performed per bank to ensure themes reflect each bank's specific user feedback.

#### Output

| File | Description |
|------|-------------|
| `data/clean/reviews_sentiment.csv` | Reviews with VADER and DistilBERT scores |
| `data/clean/reviews_final.csv` | Final dataset with sentiment and themes |
| `data/clean/sentiment_by_bank.csv` | Aggregated sentiment per bank |
| `data/clean/sentiment_by_rating.csv` | Aggregated sentiment per bank and star rating |

> Note: All CSV files are listed in `.gitignore` and are not committed to GitHub.

---

### Task 3: Database Engineering

*Coming soon.*

---

### Task 4: Insights & Recommendations

*Coming soon.*

---

## CI/CD

This project uses GitHub Actions for continuous integration. On every push to `main`, the workflow installs all dependencies from `requirements.txt` to verify the environment is reproducible.

Workflow file: `.github/workflows/unittests.yml`

---

## Data

All data files are excluded from version control via `.gitignore`. To reproduce the dataset, run:

```bash
# Scrape raw reviews
python scripts/scrape.py

# Translate and clean
python scripts/preprocess.py
```

---

## Contributing

This project follows [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) for commit messages. Work is organized into task branches (`task-1`, `task-2`, etc.) and merged into `main` via Pull Requests.

---

## Author

10 Academy KAIM9 — Week 2 Challenge  
Omega Consultancy Data Analytics Team