CS31-1-Rhinoplasty-Prediction-Studio
=====================================

An AI tool that predicts the post-operative result of rhinoplasty
(nose surgery) from a single pre-operative side-profile photo.


HOW TO START
------------
1. Double-click  CS31-1-Rhinoplasty-Prediction-Studio.exe
2. On the very first launch the app downloads a ~4 GB AI model.
   Read FIRST_LAUNCH.txt before you begin.


WHAT IS IN THIS FOLDER
----------------------
CS31-1-Rhinoplasty-Prediction-Studio.exe   The application. Double-click to run.
download_sd_model_v3.bat                   Backup model downloader (see FIRST_LAUNCH.txt).
README.txt                                 This file.
FIRST_LAUNCH.txt                           First-launch and download instructions.
_internal\                                 Program files. DO NOT delete, move or rename.

Keep every item together in the same folder. The app will not start if
the _internal folder is missing.


PHOTO TIPS FOR BEST RESULTS
---------------------------
Framing
  - Frame from the shoulders up. Do not include the chest, the full body
    or large amounts of clothing. Extra body area pulls the model's focus
    away from the face and produces blurred predictions.
  - Use a plain, uncluttered background such as a solid wall. Busy
    backgrounds add noise that lowers prediction quality.
  - Use a true side profile (90 degrees from the front) with the nose
    clearly visible.

Resolution
  - Use the highest-resolution photo you have. The prediction is produced
    at the same resolution as the input, so a sharper photo gives a
    sharper, clearer result.
  - Minimum 512 x 512 pixels. 1024 pixels or larger is noticeably better.

Format
  - One person only.
  - JPEG, PNG or WEBP.


FILE LOCATIONS
--------------
AI model :  %APPDATA%\CS31-1-Rhinoplasty-Prediction-Studio\models\sd_base\inpaint\
Log file :  %APPDATA%\CS31-1-Rhinoplasty-Prediction-Studio\cs31-rhinoplasty-prediction-studio.log

Paste either path into the File Explorer address bar to open it.


TROUBLESHOOTING
---------------
If the app cannot download the model on first launch, run
download_sd_model_v3.bat - see FIRST_LAUNCH.txt for full instructions.
