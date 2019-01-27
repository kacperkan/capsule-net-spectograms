from pathlib import Path
import logging

MODELS_PATH = Path('models')
MODELS_PATH.mkdir(exist_ok=True)

DATA_PATH = Path('data')
X_DATA_PATH = DATA_PATH / "competition_dataset.npy"
Y_DATA_PATH = DATA_PATH / "competition_dataset_labels.npy"
TEST_DATA_PATH = DATA_PATH / "competition_test.npy"

DATA_MEAN = 0.004294426923977129
DATA_STD = 0.9834690962728733
DATA_MAX = 2.624241376197074
DATA_MIN = -2.474743176461429

INITIAL_SIZE = 32

logger = logging.getLogger("spectograms")
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
