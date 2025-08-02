import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data  # Features
y = iris.target  # Labels

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Create a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Print the accuracy
print ('Accuracy: ', accuracy*100)

# *master* is the name of the branch to which commit is made. *HEAD* is pointer pointing to the current branch where commit is currently made. It also means which commit is actually being considered. So, if you have suppose 3 commits, then you can use *HEAD* to actually load any commit you want and not necessarily the latest commit made by you. 

# To see this in action, let's make a change to the file. You calculate F1 score and display it. 

# make following changes to the filea321421e26353e5272ba0810b39194574691193c

f1 = f1_score(y_test, y_pred, average='macro')
print ('F1 score: ', f1)


