MODEL_PARAMS = {
    "logistic_regression": {
        "penalty": ["l2"],
        "C": [0.01, 0.1, 1, 10],
    },
    "svc": {
        "C": [0.1, 10],
        "kernel": ["rbf"]
    },

    "decision_tree": {
        "max_depth": [5, 10, 20, None],
        "min_samples_split": [2, 5, 10]
    },
    "random_forest": {
        "n_estimators": [100, 300, 500, 1000],
        "max_depth": [10, 20, None],
    },
    "xgboost": {
        "n_estimators": [200, 300, 500, 1000],
        "max_depth": [3, 5, 10, 12],
        "learning_rate": [0.05, 0.1],
    },
}
