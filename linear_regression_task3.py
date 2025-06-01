
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
url = "https://www.kaggle.com/datasets/harishkumardatalab/housing-price-prediction"
# You need to manually download and load the dataset
# Example: df = pd.read_csv("Housing.csv")

# Dummy data for example
df = pd.DataFrame({
    'area': [1000, 1500, 1800, 2400, 3000],
    'price': [100000, 150000, 180000, 240000, 300000]
})

# Features and target
X = df[['area']]
y = df['price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Plot
plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test, y_pred, color='red')
plt.xlabel('Area')
plt.ylabel('Price')
plt.title('Linear Regression')
plt.savefig("regression_plot.png")
plt.show()
