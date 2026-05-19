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
---

## First-Time Launch Instructions (macOS)

Because `CS31Preview.app` is not signed with an Apple Developer ID, macOS may block it on first launch. Please follow the steps below to approve and open the application.

1. Drag `CS31Preview.app` into the `/Applications` folder.

2. Control-click (or right-click or double-click) the app and select **Open**.

3. A warning may appear stating that the application cannot be opened because it is from an unidentified developer. This is expected. Click **Done** to close the dialog.
<img width="1201" height="635" alt="1_mac_app_open_warning" src="https://github.com/user-attachments/assets/13b35c3e-eafd-450e-b751-8be65833c047" />

4. Open **System Settings** on your Mac.
<img width="675" height="973" alt="1 1_mac_sys_setting" src="https://github.com/user-attachments/assets/118960cb-5938-4133-9fff-fb738dcc3d6f" />

5. Navigate to:

```text id="6yw5r6"
Privacy & Security → Security
```

6. You should see a message similar to:

```text id="26utvj"
"CS31Preview" was blocked to protect your Mac
```

Click **Open Anyway**.

<img width="725" height="907" alt="1 2_mac_privacy_open_cs31" src="https://github.com/user-attachments/assets/d6b67d61-6d42-422a-902e-61e32dea3a29" />

7. A confirmation dialog will appear again. Click **Open Anyway**, then enter your Mac password if prompted.
<img width="841" height="890" alt="1 3_mac_open_cs31" src="https://github.com/user-attachments/assets/7deb6516-44c7-4890-ae85-75cff4f2c05b" />

9. During the first launch, the system need to download **Stable Diffusion base model (4GB)**  from a mirrored server, this is a one-time action, future action will be using the downloaded local copy. This process may take 5-10 minutes. *This apply for **BOTH** **MacOS** and **Windows** versions*
<img width="1014" height="644" alt="1 5_mac_win_dowload_base_model" src="https://github.com/user-attachments/assets/affc0050-1a5b-422c-85cb-318f3fb3d9f0" />


10. `CS31Preview` should now launch successfully.

<img width="1167" height="819" alt="1 4_mac_app_opened" src="https://github.com/user-attachments/assets/9fde0968-39c7-4583-8867-c13c1e1787b1" />
