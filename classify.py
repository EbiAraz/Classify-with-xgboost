import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import os
import sys
from sklearn.model_selection import train_test_split, cross_val_predict, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings


warnings.filterwarnings("ignore")

#import custom data generator
# Ensure local modules in this folder take precedence over similarly named site-packages.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_generator import portfolioDataGenerator


#set random seed
np.random.seed(42)

CONFIG = {
    "n_samples": 1000,
    "test_size": 0.2,
    "random_state": 42,
    "cv_folds": 5,
    "n_jobs": -1, #use all cpu cores
}

#initialize data generator
data_gen = portfolioDataGenerator(random_state=CONFIG["random_state"])

#Generate data with automatic train/test/split
x_train, x_test, y_train, y_test = data_gen.generate_with_split(
    n_samples=CONFIG["n_samples"], test_size=CONFIG["test_size"]
)

#Encoding

#Create label mapping
label_mapping = {"low": 0, "Medium": 1, "High": 2}
reverse_mapping = {0: "low", 1: "Medium", 2: "High"}

#Encode labels

y_train_encoded = y_train.map(label_mapping)
y_test_encoded = y_test.map(label_mapping)

#Build Baseline Model

baseline_model = xgb.XGBClassifier(
    objective="multi:softmax",
    num_class=3,
    max_depth=4,
    learning_rate=0.1,
    n_estimators=100,
    random_state=CONFIG["random_state"],
    eval_metric="mlogloss",
)

baseline_model.fit(x_train, y_train_encoded)
print("Baseline model trained!")

#Evaluate baseline
y_train_pred = baseline_model.predict(x_train)
y_test_pred = baseline_model.predict(x_test)

train_acc = accuracy_score(y_train_encoded, y_train_pred)
test_acc = accuracy_score(y_test_encoded, y_test_pred)

print(f"\nBaseline Performance:")
print(f" Training Accuracy: {train_acc:.2%}")
print(f" Testing Accuracy: {test_acc:.2%}")
print(f" Overfitting Gap: {(train_acc - test_acc):.2%}")

#Hyperparameter Tuning
param_grid = {
    "max_depth": [3, 4, 5],
    "learning_rate": [0.01, 0.1, 0.2],
    "n_estimators": [50, 100, 150],
    "min_child_weight": [1, 3, 5],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}

xgb_model = xgb.XGBClassifier(
    objective="multi:softmax",
    num_class=3,
    random_state=CONFIG["random_state"],
    eval_metric="mlogloss",
)

grid_search = GridSearchCV(
    xgb_model, param_grid, cv=3, scoring="accuracy", n_jobs=CONFIG["n_jobs"],
)

grid_search.fit(x_train, y_train_encoded)

print(f"\n Best Parameter Found:")
for param, value in grid_search.best_params_.items():
    print(f" {param}: {value}")

best_model = grid_search.best_estimator_

#Evaluate tuned model

y_train_pred_best = best_model.predict(x_train)
y_test_pred_best = best_model.predict(x_test)

train_acc_best = accuracy_score(y_train_encoded, y_train_pred_best)
test_acc_best = accuracy_score(y_test_encoded, y_test_pred_best)

print(f"\n Tuned Model Performance:")
print(f" Training Accuracy: {train_acc_best:.2%}")
print(f" Testing Accuracy: {test_acc_best:.2%}")
print(f" Improvement: {(test_acc_best - test_acc):.2%}")

#Cross Validation

cv_scores = cross_val_score(
    best_model, x_train, y_train_encoded, cv=CONFIG["cv_folds"]
)

#Detailed Analysis

#Prediction with probabilities

y_test_prob = best_model.predict_proba(x_test)

#Convert to original labels

y_test_labels = y_test_encoded.map(reverse_mapping)
y_test_pred_labels = pd.Series(y_test_pred_best).map(reverse_mapping)

#Feature importance

feature_names = data_gen.get_feature_names()
feature_importance = pd.DataFrame(
    {"feature": feature_names, "importance": best_model.feature_importances_}
).sort_values("importance", ascending=False)


print("\n Top features by importance:")
print("-" * 80)
for idx, row in feature_importance.iterrows():
    print(f" {row['feature']:<25} {row['importance']:.4f}")