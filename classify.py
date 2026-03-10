"""
classify.py - XGBoost classification using a synthetic data generator.
"""

import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report

from data_generator import DataGenerator


def train_and_evaluate(
    n_samples: int = 1000,
    n_features: int = 20,
    n_informative: int = 10,
    n_classes: int = 2,
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    random_state: int = 42,
):
    """
    Train an XGBoost classifier on synthetic data and print evaluation metrics.

    Args:
        n_samples: Number of samples to generate.
        n_features: Number of features per sample.
        n_informative: Number of informative features.
        n_classes: Number of target classes.
        n_estimators: Number of boosting rounds.
        max_depth: Maximum tree depth for base learners.
        learning_rate: Boosting learning rate (eta).
        random_state: Random seed for reproducibility.
    """
    # 1. Generate data
    generator = DataGenerator(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_classes=n_classes,
        random_state=random_state,
    )
    X_train, X_test, y_train, y_test = generator.generate()

    # 2. Select objective based on number of classes
    if n_classes == 2:
        objective = "binary:logistic"
        eval_metric = "logloss"
    else:
        objective = "multi:softmax"
        eval_metric = "mlogloss"

    # 3. Build and train the model
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        objective=objective,
        eval_metric=eval_metric,
        num_class=n_classes if n_classes > 2 else None,
        random_state=random_state,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # 4. Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)

    return model, accuracy


if __name__ == "__main__":
    print("=== Binary Classification ===")
    train_and_evaluate(n_classes=2)

    print("\n=== Multi-Class Classification (3 classes) ===")
    train_and_evaluate(
        n_classes=3,
        n_informative=12,
        n_estimators=150,
    )
