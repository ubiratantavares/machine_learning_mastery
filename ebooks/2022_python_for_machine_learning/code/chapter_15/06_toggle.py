from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# toggle between options
SCALER = "maxmin"    # "standard", "maxmin", or None
CLASSIFIER = "cart"  # "svc" or "cart"

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

# Split-out validation dataset
array = dataset.values
X = array[:,0:4]
y = array[:,4]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=1)

# Create model
if CLASSIFIER == "svc":
    model = SVC()
elif CLASSIFIER == "cart":
    model = DecisionTreeClassifier()
else:
    raise NotImplementedError

if SCALER == "standard":
    clf = Pipeline([('scaler',StandardScaler()), ('classifier',model)])
elif SCALER == "maxmin":
    clf = Pipeline([('scaler',MinMaxScaler()), ('classifier',model)])
elif SCALER == None:
    clf = model
else:
    raise NotImplementedError

# Train and test
clf.fit(X_train, y_train)
score = clf.score(X_val, y_val)
print("Validation accuracy", score)
