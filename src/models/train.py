import os
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

import mlflow
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix ,f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from configs.models_grid_search_params import MODEL_PARAMS

import tomllib

with open("./configs/config.toml", "rb") as f:
    config = tomllib.load(f)

EXPERIMENT_NAME="house_price_prediction_experiment"
TRAIN_CONFUSION_MATRIX_PATH = config["paths"]["train_confusion_matrix"]
VALIDATION_CONFUSION_MATRIX_PATH = config["paths"]["validation_confusion_matrix"]
TEST_CONFUSION_MATRIX_PATH = config["paths"]["test_confusion_matrix"]
MLFLOW_DB_MODEL_PATH = config["paths"]["mlflow_db"]
MODELS_DIR = config["paths"]["models_dir"]

TRAIN_PATH = config["paths"]["train_path"]
VALIDATION_PATH = config["paths"]["validation_path"]
TEST_PATH = config["paths"]["test_path"]


class Evaluator:
    def evaluate(self, y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro")
        report = classification_report(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        return acc, f1, report, cm

    def save_confusion_matrix(self, cm, filename):
        plt.figure()
        sns.heatmap(cm, annot=True, fmt="d") 
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.savefig(filename)
        plt.close()     


class ModelTrainer:
    def __init__(self, train_path, val_path, experiment_name, model_dir=MODELS_DIR):
        # data loading
        train_data = pd.read_csv(train_path)
        val_data = pd.read_csv(val_path)

        self.X_train = train_data.drop(columns=["target"])
        self.y_train = train_data["target"]
        self.X_val = val_data.drop(columns=["target"])
        self.y_val = val_data["target"]

        # evaluator
        self.evaluator = Evaluator()

        # models and tracking
        self.model_dir = model_dir
        self.best_model = None
        self.best_score = -1
        self.best_name = None

        self.candidate_models = {
            "logistic_regression": LogisticRegression(max_iter=25_000, random_state=42),
            "decision_tree": DecisionTreeClassifier(random_state=42),
            "random_forest": RandomForestClassifier(random_state=42, n_jobs=-1),
            "xgboost": XGBClassifier(random_state=42,n_jobs=-1,eval_metric="logloss"),
            # "svc": SVC(),
        }
        
        # setup mlflow
        mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_MODEL_PATH}")
        mlflow.set_experiment(experiment_name)

    def train_all(self, model_params_grid):
        for name, model in self.candidate_models.items():
            print(f"\nTraining {name} ...")

            params = model_params_grid[name]
            run_name = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            with mlflow.start_run(run_name=run_name):
                grid = GridSearchCV(
                    estimator=model,
                    param_grid=params,
                    cv=3,
                    scoring="f1_macro",
                    n_jobs=-1
                )

                grid.fit(self.X_train, self.y_train)

                best_model = grid.best_estimator_

                train_pred = best_model.predict(self.X_train)
                val_pred = best_model.predict(self.X_val)

                train_acc, train_f1, train_rep, train_cm = self.evaluator.evaluate(self.y_train, train_pred)
                val_acc, val_f1, val_rep, val_cm = self.evaluator.evaluate(self.y_val, val_pred)

                mlflow.log_params(grid.best_params_)
                mlflow.log_metrics({
                    "train_acc": train_acc,
                    "val_acc": val_acc,
                    "train_macro_f1": train_f1,
                    "val_macro_f1": val_f1,
                    "cv_f1_macro": grid.best_score_
                })

                mlflow.log_text(train_rep, "train_report.txt")
                mlflow.log_text(val_rep, "val_report.txt")

                self.evaluator.save_confusion_matrix(train_cm, TRAIN_CONFUSION_MATRIX_PATH)
                self.evaluator.save_confusion_matrix(val_cm, VALIDATION_CONFUSION_MATRIX_PATH)

                mlflow.sklearn.log_model(best_model, name)

                print(f"{name} val_acc: {val_acc:.4f}")
                print(f"{name} val_macro_f1: {val_f1:.4f}")

                # track best
                if val_f1 > self.best_score:
                    self.best_score = val_f1
                    self.best_model = best_model
                    self.best_name = name

        return self.best_model

    def save_best(self):

        os.makedirs(self.model_dir, exist_ok=True)
        # remove latest since current one is the latest
        latest_path = os.path.join(self.model_dir, "best_model_latest.pkl")
        if os.path.exists(latest_path):
            os.remove(latest_path)

        existing_versions = [
            f for f in os.listdir(self.model_dir)
            if f.startswith("best_model_") and f.endswith(".pkl") and f != "best_model_latest.pkl"
        ]

        version = len(existing_versions)
        version_path = os.path.join(self.model_dir,f"best_model_{version}.pkl")

        joblib.dump(self.best_model, latest_path)
        joblib.dump(self.best_model, version_path)

        print("\nSaved best model:")
        print("Model:", self.best_name)
        print("Score:", self.best_score)
        print("Latest:", latest_path)
        print("Versioned:", version_path)



if __name__ == "__main__":

    trainer = ModelTrainer(
        train_path=TRAIN_PATH,
        val_path=VALIDATION_PATH,
        experiment_name=EXPERIMENT_NAME,
        model_dir=MODELS_DIR
    )

    best_model = trainer.train_all(MODEL_PARAMS)
    trainer.save_best()