## ScamShield

A web application to detect and advise users on potential phishing/scam attempts.

CPE 595 Group 13: Kofwana Lawson, Megan Yim, Aditya Pradeep

---

## How it fits together

```
Browser  ->  Express app (Node, port 3000)  ->  Model service (Python FastAPI, port 8000)
              form + result pages                 loads ml/models/baseline_app.joblib
```

The Express app posts the subject and content to the model service's `/predict` endpoint and renders the result.

## Tech Stack

- Web app: Node.js, Express, Handlebars (express-handlebars)
- ML: Python, pandas, scikit-learn, FastAPI

## Project structure

```
app.js, routes/, views/, public/   Express web app
ml/
  data/spam_email_dataset.csv      dataset
  data/splits.csv                  fixed 70/15/15 train/val/test split (seed 42)
  train_baseline.py                logistic regression baseline on 3 feature sets
  serve.py                         FastAPI model service used by the web app
  results/                         metrics and confusion matrices
  notebooks/                       EDA and experiments go here
  models/                          trained models (not committed, rebuild with train_baseline.py)
```

---

## Setup

### 1. Prereqs

- Node.js v18 or higher
- Python 3.10 or higher

### 2. Clone the repo

```
git clone https://github.com/chadybo/CPE595WS-Project-ScamShield.git
cd CPE595WS-Project-ScamShield
```

### 3. Install dependencies

```
npm install

cd ml
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Create a .env file

Copy `.env.example` to `.env` in the project root:

```env
PORT=3000
ML_URL=http://localhost:8000
```

### 5. Train the baseline

```
cd ml
python train_baseline.py
```

This prints metrics, writes `ml/results/`, and saves the model the app uses.

### 6. Start both services (two terminals)

```
# Terminal 1
cd ml
uvicorn serve:app --port 8000

# Terminal 2 (project root)
npm start
```

Visit `http://localhost:3000` in your browser.

---

## Baseline results (validation set, logistic regression)

| Feature set | What it uses | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| A_all | text + metadata + sender domain | 1.000 | 1.000 | 1.000 | 1.000 |
| B_no_leaky_meta | body text + metadata, no subject or sender | 0.999 | 1.000 | 0.998 | 0.999 |
| C_metadata_only | numeric metadata only | 0.884 | 0.854 | 0.856 | 0.855 |
| app_text | subject + body text (what the web form sends) | 1.000 | 1.000 | 1.000 | 1.000 |

Note: the dataset is synthetic. `subject` and `sender_domain` each separate the classes perfectly, and the body text uses obvious spam words, so text models hit about 100%. Feature set C is the meaningful place to compare Naive Bayes and SVM against the baseline.

Run `python train_baseline.py --test` only for the final report, so the test set stays untouched until then.

---

## Working on the repo

- Branch off `main` for each task: `feature/<short-name>` (e.g. `feature/naive-bayes`, `feature/results-page`)
- Open a pull request into `main` and have one teammate look it over before merging
- Pull `main` often to avoid big merge conflicts
