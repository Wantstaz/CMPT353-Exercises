import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

# Read the labelled data
labelled_data = pd.read_csv(sys.argv[1])
# Read the unlabelled data
unlabelled_data = pd.read_csv(sys.argv[2])
X_unlabelled = unlabelled_data.iloc[:, 2:].values


X_labelled = labelled_data.iloc[:, 2:].values
y_labelled = labelled_data['city'].values
scaler = StandardScaler()
X_labelled = scaler.fit_transform(X_labelled)

X_train, X_valid, y_train, y_valid = train_test_split(X_labelled, y_labelled, test_size=0.2, random_state=42)

# Train a K-Nearest Neighbors classifier
knn_model = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=12)
)
knn_model.fit(X_train, y_train)

# Score of the model on the validation set
score = knn_model.score(X_valid, y_valid)
print("Model Score:", score)


# Normalize the unlabelled features and predict
X_unlabelled = scaler.transform(X_unlabelled)
predictions = knn_model.predict(X_unlabelled)

pd.Series(predictions).to_csv(sys.argv[3], index=False, header=False)