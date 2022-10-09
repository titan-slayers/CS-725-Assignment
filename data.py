from sklearn.datasets import make_regression
from matplotlib import pyplot
from sklearn.model_selection import train_test_split

X, y = make_regression(n_samples=40000, n_features=1, noise=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

X_train.tofile('train_X.csv', sep = ',')
y_train.tofile('train_Y.csv', sep = ',')
X_test.tofile('test_X.csv', sep = ',')
y_test.tofile('test_Y.csv', sep = ',')