import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
data = pd.read_csv("dataset.csv")

data = np.array(data)
X = data[:, 0:-1]  # Features
y = data[:, -1]    # Target variable

y = y.astype('int')
X = X.astype('int')

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train the model
rfc = RandomForestClassifier(random_state=0)
rfc.fit(X_train, y_train)

# Save the trained model using pickle
with open('model.pkl', 'wb') as model_file:
    pickle.dump(rfc, model_file)