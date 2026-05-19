# CS31 Project Report Draft

## Title

Profile-to-Profile Rhinoplasty Outcome Prediction from Paired Facial Images

## Dataset

- Source: WhatsApp-exported rhinoplasty image collection stored in [CS31_Rhioplasty_Outcome_Prediction](/Applications/CS31/CS31_Rhioplasty_Outcome_Prediction)
- Structure: single-canvas paired images where the left half is pre-operative and the right half is post-operative
- Additional source containers: ZIP archives included in the same directory
- Cleaning contribution:
  - unified indexing across extracted JPEG files and ZIP members
  - duplicate filtering using filename and perceptual hash
  - automatic split into pre-op and post-op halves
  - fixed train/val/test split export

## Methodology

- Input modality: profile-only paired image translation
- Implemented baselines:
  - Autoencoder
  - Pix2Pix
  - CycleGAN
  - Lightweight diffusion feasibility run
- Production focus:
  - benchmark all four methods
  - promote Pix2Pix and the best benchmark companion model to the final system

## Evaluation

- Whole-image metrics:
  - SSIM
  - LPIPS
  - FID
- Local nasal-region metrics:
  - ROI SSIM
  - ROI LPIPS
- Qualitative artifacts:
  - prediction triplets
  - benchmark table
  - manual review template

## Web Application

- Backend: FastAPI + SQLite history
- Frontend: React + Vite
- Features:
  - upload paired canvas or pre-op image
  - predict post-op output
  - compare pre-op, real post-op, and generated post-op
  - view benchmark table
  - inspect local history

## Limitations

- No real cost labels
- No real text or notes metadata
- No patient-level identifier beyond image-level assumptions
- All current conclusions are restricted to image-based outcome prediction

## Future Work

- Integrate real `cases.csv` labels for cost modeling
- Integrate real `notes.csv` text fields for NLP conditioning
- Add stronger ROI alignment and clinician annotations
- Extend from profile-only to multi-view modeling

## Ethics

- Research prototype only
- Not for clinical decision-making
- Dataset provenance and duplication issues are explicitly documented

