import joblib
import pandas as pd
from src.config import FROZEN_MODEL_PATH, TRAIN_DATA_PATH
from src.data import load_data

pipeline = joblib.load(FROZEN_MODEL_PATH)
X_dev, y_dev, _ = load_data(TRAIN_DATA_PATH, require_target=True, drop_leakage=True)

prep = pipeline.named_steps["preprocessing"]
X_proc = prep.transform(X_dev[:10])
print("X_proc shape:", X_proc.shape)

try:
    feature_names = prep.get_feature_names_out()
    print("Feature names (len={}):".format(len(feature_names)))
    for i, name in enumerate(feature_names):
        print(f"  {i}: {name}")
except Exception as e:
    print("get_feature_names_out error:", e)
