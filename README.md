# CS31 Rhinoplasty Outcome Prediction

Local research prototype for paired profile-to-profile rhinoplasty outcome prediction.

## Project Layout

- `ml/`: dataset indexing, pair preparation, training, evaluation
- `backend/`: FastAPI API, inference, SQLite history
- `frontend/`: React + Vite web application
- `data/`: manifests, splits, annotation templates
- `artifacts/`: prepared pairs, evaluation outputs, prediction images
- `reports/`: English report draft and review templates

## Dataset Assumptions

- Source directory: `CS31_Rhioplasty_Outcome_Prediction/`
- Each paired image is stored as a single canvas:
  - left half = pre-op profile
  - right half = post-op profile
- ZIP files inside the source directory are part of the official dataset.

## Python Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ml.index_dataset
python -m ml.prepare_pairs
python -m ml.train_outcome --model pix2pix --epochs 1 --limit 32
python -m ml.evaluate_outcome --model pix2pix --limit 16
python -m backend.serve
```

## Frontend Quick Start

```bash
source .nodeenv/bin/activate
cd frontend
npm install
npm run dev
```

## Notes

- `cost` and `NLP` schemas are scaffolded in `data/annotation_template.csv`, `data/cases.csv`, and `data/notes.csv`.
- They are intentionally not used for formal modeling until real labels are added.

