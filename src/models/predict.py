import joblib
import pandas as pd

from src.models.train import Evaluator
import tomllib

with open("./configs/config.toml", "rb") as f:
    config = tomllib.load(f)

MODEL_PATH = config["prediction"]["model_path"]
TEST_PATH = config["prediction"]["test_data_path"]

def predict(MODEL_PATH, TEST_PATH):
    # Load the model
    model = joblib.load(MODEL_PATH)

    # Load the test data
    Test_data = pd.read_csv(TEST_PATH)
    X_test = Test_data.drop(columns=["target"])
    y_test = Test_data["target"]

    # predict
    pred = model.predict(X_test)
    
    # evaluate
    evaluator = Evaluator()
    acc, f1, report, cm = evaluator.evaluate(y_test, pred)
    print(f"Accuracy: {acc}")
    print(f"Macro F1 Score: {f1}")
    print("Classification Report:")
    print(report)
    print("Confusion Matrix:")
    print(cm)

    return pred


if __name__ == "__main__":
    predict(MODEL_PATH, TEST_PATH)