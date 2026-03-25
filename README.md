# Sleep vs Energy — Polynomial Regression Demo

This is a minimal machine learning project that models the relationship between **hours of sleep** and **energy levels** using **polynomial regression**.

The dataset is tiny and hand‑crafted, but intentionally non‑linear: energy increases with sleep, peaks around 7 hours, then gradually decreases. A simple linear model cannot capture this shape, so a quadratic model is used instead.

## How it works

1. A small dataset of `(hours_slept → energy_level)` pairs is defined.
2. The input features are expanded using `PolynomialFeatures(degree=2)` to allow the model to learn a curved relationship.
3. A `LinearRegression` model is trained on the transformed features.
4. The model's R² score is computed.
5. The script predicts the expected energy level for 7 hours of sleep.

## Files

- `ml_minimal.py` — main script containing the full pipeline

## Example Output

Model score: 0.932
Prediction for 7 hours of sleep: 8.676

## Requirements

- Python 3.x
- NumPy
- scikit-learn

Install dependencies:

pip install numpy scikit-learn

## Running the script

python ml-minimal.py

## Why polynomial regression?

The dataset has a clear peak at 7 hours of sleep.  
A straight line cannot model this behaviour, but a quadratic curve can.  
This makes the project a simple demonstration of when and why polynomial regression is useful.
