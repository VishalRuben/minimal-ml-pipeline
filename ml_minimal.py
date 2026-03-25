import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


def main():
    # 1: Create a tiny dataset (x, y)
    # x = 2D array of hours slept
    # y = 1D array of energy levels (score from 1 to 10)
    x = np.array([[1], [2], [3], [6], [7], [8], [9], [11]])
    y = np.array([1, 2, 3, 8, 10, 9, 8, 6])

    # 2: Create and train the model
    poly = PolynomialFeatures(degree=2)
    x_poly = poly.fit_transform(x)
    model = LinearRegression()
    model.fit(x_poly, y)

    # 3: Evaluate the model
    score = model.score(x_poly, y)

    # 4: Make a prediction
    prediction = model.predict(poly.transform([[7]]))

    # 5: Print the results
    print(f"Model score: {score:.3f}")
    print(f"Prediction for 7 hours of sleep: {prediction[0]:.3f}")


if __name__ == "__main__":
    main()
