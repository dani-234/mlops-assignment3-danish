"""Generate the canonical deterministic classification dataset."""

from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = ["feature_a", "feature_b", "feature_c", "feature_d"]
TARGET = "target"


def generate(path: str | Path = "data/raw_dataset.csv", rows: int = 400, seed: int = 42) -> Path:
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(rows, len(FEATURES)))
    target = (features[:, 0] + features[:, 1] * 0.8 - features[:, 2] * 0.3 + rng.normal(0, 0.2, rows) > 0).astype(int)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(features, columns=FEATURES).assign(**{TARGET: target}).to_csv(output, index=False)
    return output


if __name__ == "__main__":
    generate()

