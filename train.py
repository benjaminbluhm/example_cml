import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

# Read in data
df_train = pd.read_csv('data/titanic_train.csv', header=None)
df_validation = pd.read_csv('data/titanic_validation.csv', header=None)

# Create features and labels
X_train = df_train.loc[:, 1:]
y_train = df_train.loc[:, 0]
X_val = df_validation.loc[:, 1:]
y_val = df_validation.loc[:, 0]

# Fit the model
clf = LogisticRegression(random_state=2, max_iter=1000)
clf.fit(X_train, y_train)

# Accuracy
acc = clf.score(X_val, y_val)
print(acc)
with open("metrics.txt", 'w') as outfile:
        outfile.write("Accuracy: " + str(acc) + "\n")


# Plot it
disp = plot_confusion_matrix(clf, X_val, y_val, normalize='true',cmap=plt.cm.Blues)
plt.savefig('confusion_matrix.png')