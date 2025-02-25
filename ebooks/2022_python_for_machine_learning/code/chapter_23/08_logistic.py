from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_openml

X, y = fetch_openml(data_id=42437, return_X_y=True, as_frame=False)
clf = LogisticRegression(random_state=0).fit(X, y)
print(clf.score(X,y)) # accuracy
print(clf.coef_)      # coefficient in logistic regression
