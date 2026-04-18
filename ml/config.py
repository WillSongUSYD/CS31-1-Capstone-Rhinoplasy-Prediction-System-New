from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = REPO_ROOT / "CS31_Rhioplasty_Outcome_Prediction"
DATA_DIR = REPO_ROOT / "data"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
MODELS_DIR = REPO_ROOT / "models"
OUT_DIR = REPO_ROOT / "out"

RAW_CACHE_DIR = DATA_DIR / "raw_cache"
PAIR_FULL_DIR = ARTIFACTS_DIR / "dataset" / "pairs_full"
PAIR_256_DIR = ARTIFACTS_DIR / "dataset" / "pairs_256"
PAIR_ALIGNED_DIR = ARTIFACTS_DIR / "dataset" / "pairs_aligned_256"
MASK_DIR = ARTIFACTS_DIR / "dataset" / "masks_256"
EVAL_DIR = ARTIFACTS_DIR / "eval"
PREDICTIONS_DIR = ARTIFACTS_DIR / "predictions"

MANIFEST_PATH = DATA_DIR / "manifest.csv"
SPLITS_PATH = DATA_DIR / "splits.csv"
ANNOTATION_TEMPLATE_PATH = DATA_DIR / "annotation_template.csv"
CASES_TEMPLATE_PATH = DATA_DIR / "cases.csv"
NOTES_TEMPLATE_PATH = DATA_DIR / "notes.csv"
SUMMARY_PATH = DATA_DIR / "dataset_summary.json"
BENCHMARK_PATH = EVAL_DIR / "benchmark.csv"

DB_PATH = DATA_DIR / "history.sqlite3"

DEFAULT_SPLIT_SEED = 31
DEFAULT_IMAGE_SIZE = 256
DEFAULT_TEST_RATIO = 0.1
DEFAULT_VAL_RATIO = 0.1
DEFAULT_NEAR_DUPLICATE_THRESHOLD = 4


def ensure_directories() -> None:
    for path in [
        DATA_DIR,
        ARTIFACTS_DIR,
        MODELS_DIR,
        OUT_DIR,
        RAW_CACHE_DIR,
        PAIR_FULL_DIR,
        PAIR_256_DIR,
        PAIR_ALIGNED_DIR,
        MASK_DIR,
        EVAL_DIR,
        PREDICTIONS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)

