# Import required packages
import os
import pickle
from sklearn.neighbors import KNeighborsClassifier
from average_precision import *

# Load the features for the dataset and queries
print("[INFO] Loading data..")
with open("features/queries.npy", "rb") as f:
    query_data = np.load(f)
with open("features/dataset.npy", "rb") as f:
    dataset = np.load(f)

# Load the dataset and query labels
with open("features/query_labels.pkl", "rb") as f:
    query_meta = pickle.load(f)
with open("features/data_labels.pkl", "rb") as f:
    data_labels = pickle.load(f)

# Convert labels to numpy arrays
labels = np.asarray([x[1] for x in data_labels])
ground_truth = np.asarray([x[1] for x in query_meta])

# Train the classifier
print("[INFO] Training..")
knn = KNeighborsClassifier(n_neighbors=5, metric="minkowski")
knn = knn.fit(dataset, labels)
neighbors = knn.kneighbors(query_data, return_distance=False)

# Get the labels obtained by classifier
predictions = []
for i in range(len(neighbors)):
    neighbors_class = [data_labels[j][1] for j in neighbors[i]]
    predictions.append(neighbors_class)

# Evaluate the model
print("[INFO] Evaluating..")
ground_truth = [x[1] for x in query_meta]
p_1 = mpk(ground_truth, predictions, 1)
p_5 = mpk(ground_truth, predictions, 5)

print('P@1=', p_1)
print('P@5=', p_5)
map = mAP(ground_truth, predictions)
print('mAP=', map)
