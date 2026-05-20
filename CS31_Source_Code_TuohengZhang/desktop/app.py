"""CS31-1-Rhinoplasty-Prediction-Studio entrypoint.

Order matters:

1. ``install_environment()`` sets the four ``CS31_*`` env vars + ``HF_ENDPOINT``
   BEFORE any ``backend.*`` / ``ml.*`` import — those modules read
   configuration eagerly at import time.
2. Logging setup.
3. QApplication + MainWindow.

The heavy ``backend.inference_sd`` import is deferred to ``InferenceWorker``
(Phase 3) so app startup stays snappy (diffusers + torch pull ~800 MB of
Python code; lazy loading puts that on the first predict, not launch).
"""
from __future__ import annotations

import logging
import sys

from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication

# MUST happen before any project-code import.
from desktop.core.config import install_environment

install_environment()

# Safe to import project code now.
from desktop.core.paths import bundle_root, is_sd_base_present  # noqa: E402
from desktop.main_window import MainWindow  # noqa: E402
from desktop.widgets.onboarding_dialog import OnboardingDialog  # noqa: E402


logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    """File + stderr logging. Log file lives under Application Support so
    users can email it to us for debugging without needing Terminal access.
    """
    from desktop.core.paths import user_support_dir
    log_file = user_support_dir() / "cs31-rhinoplasty-prediction-studio.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logger.info("Logging initialised → %s", log_file)


def _load_stylesheet(app: QApplication) -> None:
    """Apply bundled QSS. Non-fatal if missing (e.g. during very early
    dev runs before assets/ is populated) - the app still renders with
    default styling."""
    qss_path = bundle_root() / "assets" / "style.qss"
    if qss_path.exists():
        app.setStyleSheet(qss_path.read_text(encoding="utf-8"))
        logger.info("stylesheet loaded from %s", qss_path)
    else:
        logger.warning("stylesheet missing: %s", qss_path)


def main() -> int:
    _setup_logging()

    app = QApplication(sys.argv)
    app.setApplicationName("CS31-1-Rhinoplasty-Prediction-Studio")
    app.setApplicationDisplayName("CS31-1-Rhinoplasty-Prediction-Studio")
    app.setOrganizationName("CS31")

    # Optional icon (Phase 5 will ship a real .icns).
    icon_path = bundle_root() / "assets" / "icon.icns"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    _load_stylesheet(app)

    # First-launch onboarding: if the 4GB SD base isn't present, block
    # the main window behind a download dialog. Users who cancel here
    # exit the app (there's nothing meaningful to do without the base).
    if not is_sd_base_present():
        logger.info("SD base missing — running onboarding download flow")
        dlg = OnboardingDialog()
        result = dlg.exec()
        if result != OnboardingDialog.DialogCode.Accepted:
            logger.info("onboarding cancelled; exiting")
            return 0

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
