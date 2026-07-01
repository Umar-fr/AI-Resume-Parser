# Redrob Candidate Ranker

Deterministic ranker for the Hack2Skill / Redrob AI Intelligent Candidate Discovery & Ranking Challenge (Track 1).

The job description explicitly warns that keyword matching is a trap. Looking at the sample submission they provide confirms it — an HR Manager ranks above an ML Engineer purely because they listed more AI skill names. This ranker ignores skill counts entirely and scores what candidates actually did: their career history, title, real experience computed from job durations, company type, and whether they're actually reachable.

---

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/Umar-fr/AI-Resume-Parser.git
cd AI-Resume-Parser
```

### 2. Set up a virtual environment (recommended)

This project is tested with Python 3.11. Python 3.13 can trigger source builds for pandas and fail with dependency-install errors, so use 3.11 if possible.

```bash
# Windows with Python 3.11
py -3.11 -m venv venv
venv\Scripts\activate

# macOS / Linux
python3.11 -m venv venv
source venv/bin/activate
```

If pip reports `No space left on device`, create the virtual environment on a drive with more free space and point the temp directory there:

```powershell
New-Item -ItemType Directory -Force -Path D:\tmp\pip | Out-Null
$env:TEMP = 'D:\tmp\pip'
$env:TMP = 'D:\tmp\pip'
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the candidate data

The full dataset (`candidates.jsonl`, ~465 MB) is not in the repo. Download it from the Hack2Skill challenge portal and place it at:

```
data/candidates.jsonl
```

The bundled `data/sample_candidates.json` (50 candidates) works for smoke tests without the full file.

---

## Running the ranker

```bash
python rank.py --candidates data/candidates.jsonl --out outputs/submission.csv
```

Typical runtime is around **22 seconds** on a standard laptop (CPU only, no GPU needed, no network calls).

To test against the sample data first:

```bash
python rank.py --candidates data/sample_candidates.json --out outputs/sample_out.csv
```

### Validate the output

```bash
python validate_submission.py outputs/submission.csv
```

This checks the format requirements from the submission spec: exactly 100 rows, correct columns, non-increasing scores, no duplicate ranks or IDs.

---

## Sandbox

Sandbox link: https://ai-resume-parser-66wb8txeanshqvaj9b32jf.streamlit.app/
In the data folder there is a pre-loaded file called sample_candidates.json which is used to reproduce the working of this project. The output is in the outputs folder which can also be downloaded directly from the application interface after ranking.

To run this locally: 
```bash
streamlit run app.py
```

Open this URL to access the app locally on browser: http://localhost:8502

## Running tests

```bash
pytest test_pipeline.py -v
```

Three tests: loader reads the sample JSON array, honeypot detector flags zero-duration expert skills, and end-to-end ranking on the sample produces sorted results with valid reasoning strings.

---

## Cleaning up

To remove Python cache files:

```bash
# Remove all __pycache__ directories and .pyc files
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete
find . -name "*.pyo" -delete

# Remove pytest cache
rm -rf .pytest_cache
```

To wipe the virtual environment and start fresh:

```bash
deactivate
rm -rf venv
```

---

## How scoring works

**Base fit score** (weighted sum, all components 0.0–1.0):

| Component | Weight | What it measures |
|---|---|---|
| Career history | 35% | Template tier 1–5 based on actual work described |
| Title fit | 15% | Whether the current title is genuinely AI/ML-relevant |
| Real experience | 15% | Years computed from career durations, not the self-reported field |
| Company type | 10% | Product company vs. IT services vs. fictional filler |
| Trusted skills | 15% | JD-relevant skills weighted by proficiency, duration, and endorsements |
| Location | 10% | Pune/Noida preferred; other Indian cities with relocation willingness |

**Behavioral multiplier** applied on top (0.50–1.00x): combines platform activity recency, recruiter response rate, interview completion rate, open-to-work flag, notice period, and account verification. The JD says to down-weight unreachable candidates — not disqualify them — so this is a multiplier, not a hard cut.

**Honeypots** are hard-excluded before scoring: any candidate with two or more "expert" skills showing zero months of actual use gets dropped entirely.

### The career history finding

The dataset's `career_history[].description` field isn't free text — there are exactly 23 unique description strings reused across thousands of candidates. Rather than run keyword matching over recycled boilerplate, every template was read and manually ranked 1–5. Tier 5 templates describe work that nearly mirrors the JD word-for-word: migrating keyword search to embedding-based retrieval on a candidate corpus, building a recruiter-facing RAG ranking pipeline, reporting NDCG@10 as an eval metric. Scoring a candidate's career history is then a table lookup, which is fast, auditable, and easy to explain.

---

## File overview

```
rank.py                  — entry point, writes the submission CSV
scoring.py               — composite scoring and ranking logic
features.py              — individual feature functions
career_templates.py      — manually tiered template lookup table
constants.py             — title/company classification sets
reasoning.py             — generates the reasoning column from score evidence
load_candidates.py       — streaming JSONL loader
explore.py               — exploratory analysis scripts (not used at inference)
validate_submission.py   — format checker from the challenge bundle
test_pipeline.py         — smoke tests
```

---

## Notes

AI tools (Claude, ChatGPT) were used throughout development for analysis, implementation, and review. The ranking logic itself is deterministic Python and makes no AI API calls at inference time.
