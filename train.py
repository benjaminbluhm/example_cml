import pandas as pd
from sklearn.linear_model import LogisticRegression
from pickle import dump

# Read in data
df_train = pd.read_csv('data/titanic_train.csv', header=None)

# Create features and labels
X_train = df_train.loc[:, 1:]
y_train = df_train.loc[:, 0]

# Fit the model
clf = LogisticRegression(random_state=20, max_iter=1000)
clf.fit(X_train, y_train)

# Save down trained model
dump(clf, open( "trained_model.pkl", "wb" ) )
