# Classify-with-xgboost

How to classify with the XGBoost library using a synthetic data generator.

## Overview

This project demonstrates binary and multi-class classification using [XGBoost](https://xgboost.readthedocs.io/), with synthetic datasets produced by a reusable `DataGenerator` class backed by scikit-learn.

## Files

| File | Description |
|------|-------------|
| `data_generator.py` | `DataGenerator` class – creates synthetic train/test splits |
| `classify.py` | Trains an XGBoost classifier and prints accuracy & classification report |
| `requirements.txt` | Python dependencies |

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python classify.py
```

### Example output

```
=== Binary Classification ===
Dataset generated: 1000 samples, 20 features, 2 classes.
Train size: 800, Test size: 200

Accuracy: 0.9350

Classification Report:
              precision    recall  f1-score   support
           0       0.95      0.92      0.93        95
           1       0.93      0.95      0.94       105
    accuracy                           0.94       200

=== Multi-Class Classification (3 classes) ===
Dataset generated: 1000 samples, 20 features, 3 classes.
Train size: 800, Test size: 200

Accuracy: 0.7950
...
```

## Using `DataGenerator` in your own code

```python
from data_generator import DataGenerator
import xgboost as xgb

gen = DataGenerator(n_samples=2000, n_features=15, n_informative=8, n_classes=2)
X_train, X_test, y_train, y_test = gen.generate()

model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1)
model.fit(X_train, y_train)
print("Accuracy:", model.score(X_test, y_test))
```
