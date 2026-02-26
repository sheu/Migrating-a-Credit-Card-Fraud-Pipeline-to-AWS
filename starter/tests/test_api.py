import os
from pathlib import Path

import requests
import pytest
import pandas as pd
from sklearn.preprocessing import StandardScaler

BASE_URL = os.getenv("API_BASE_URL", "http://fraud-pi-alb-cepdzmis3xam-1261845563.us-east-1.elb.amazonaws.com")


def _find_dataset_path() -> Path | None:
    """
    Prefer DATASET_PATH env var, otherwise try common repo-relative locations.
    """
    env_path = os.getenv("DATASET_PATH")
    if env_path:
        p = Path(env_path).expanduser()
        return p if p.exists() else None

    # Try likely locations relative to this test file
    here = Path(__file__).resolve()
    candidates = [
        # starter/data/...
        here.parents[1] / "data" / "credit_card_transaction_data_labeled.csv",
        # data/...
        here.parents[2] / "data" / "credit_card_transaction_data_labeled.csv",
        # starter/data might be elsewhere depending on run cwd
        Path.cwd() / "starter" / "data" / "credit_card_transaction_data_labeled.csv",
        Path.cwd() / "data" / "credit_card_transaction_data_labeled.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


dataset_path = _find_dataset_path()
if dataset_path is None:
    pytest.skip(
        "Dataset CSV not found. Set DATASET_PATH or place credit_card_transaction_data_labeled.csv under starter/data/ or data/.",
        allow_module_level=True,
    )

df = pd.read_csv(dataset_path)

# Select a sample row for testing (exclude label)
sample_row = df.drop(columns=["Class"]).iloc[0]

# Fit scaler on full feature set (portable)
scaler = StandardScaler()
scaler.fit(df.drop(columns=["Class"]))

sample_df = pd.DataFrame([sample_row])
sample_scaled = scaler.transform(sample_df)

columns = df.drop(columns=["Class"]).columns.tolist()
sample_scaled_df = pd.DataFrame(sample_scaled, columns=columns)
sample_scaled_list = sample_scaled_df.iloc[0].tolist()


@pytest.fixture
def sample_input():
    return {"features": sample_scaled_list}


def test_predict(sample_input):
    response = requests.post(f"{BASE_URL}/predict/", json=sample_input, timeout=30)
    assert response.status_code == 200, response.text
    body = response.json()
    assert "prediction" in body
    assert isinstance(body["prediction"], (int, float))