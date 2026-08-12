from pathlib import Path
import joblib
import pandas as pd

BASE = Path(__file__).resolve().parent
pipeline = joblib.load(BASE / "outputs" / "best_complete_pipeline.joblib")

raw = pd.DataFrame([
    {"pclass": 3, "age": 30, "sibsp": 0, "parch": 0,
     "fare": 8.05, "sex": "male", "embarked": "S"},
    {"pclass": 1, "age": 35, "sibsp": 1, "parch": 0,
     "fare": 100.0, "sex": "female", "embarked": "C"}
])

print("Reloaded:", type(pipeline).__name__)
print("Raw-input predictions:", pipeline.predict(raw).tolist())
