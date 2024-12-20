from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

# Load the boston dataset
boston = load_boston()
X = boston.data
y = boston.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Create a Linear regression model
linear_reg = LinearRegression()

# Fit the model to the training data
linear_reg.fit(X_train, y_train)

# Predict the output for test data
y_pred = linear_reg.predict(X_test)

# Print the coefficients of the model
print("Coefficients: ", linear_reg.coef_)

# Print the intercept of the model
print("Intercept: ", linear_reg.intercept_)

# Print the mean squared error of the model
from sklearn.metrics import mean_squared_error
print("Mean Squared Error: ", mean_squared_error(y_test, y_pred))
