from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

# Split-out validation dataset
array = dataset.values
X = array[:,0:4]
y = array[:,4]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=1)

# Train
clf = Pipeline([('scaler',StandardScaler()), ('classifier',SVC())])
clf.fit(X_train, y_train)

# Test
score = clf.score(X_val, y_val)
print("Validation accuracy", score)
