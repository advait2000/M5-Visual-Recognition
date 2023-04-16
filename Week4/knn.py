# Import required packages
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import label_binarize
from average_precision import *

# Load the features for the dataset and queries
print("[INFO] Loading data..")
with open("features/queries.npy", "rb") as f:
    query_data = np.load(f)
with open("features/dataset.npy", "rb") as f:
    dataset = np.load(f)

# Load the dataset and query labels
with open("features/query_labels.pkl", "rb") as f:
    query_labels = pickle.load(f)
with open("features/data_labels.pkl", "rb") as f:
    data_labels = pickle.load(f)

# Convert labels to numpy arrays
labels = np.asarray([x[1] for x in data_labels])
ground_truth = np.asarray([x[1] for x in query_labels])

# initialize the values of k for our k-Nearest Neighbor classifier along with the
# list of accuracies for each value of k
kVals = range(1, 30, 2)
accuracies = []

# loop over various values of `k` for the k-Nearest Neighbor classifier
print("[INFO] Choosing Best K")
for k in range(1, 30, 2):
    print("[INFO] Training: K={}".format(k))
    knn = KNeighborsClassifier(n_neighbors=k, metric="minkowski")
    knn = knn.fit(dataset, labels)
    neighbors = knn.kneighbors(query_data, return_distance=False)
    predictions = []
    for i in range(len(neighbors)):
        neighbors_class = [data_labels[j][1] for j in neighbors[i]]
        predictions.append(neighbors_class)
    ground_truth = [x[1] for x in query_labels]
    map = mAP(ground_truth, predictions)
    accuracies.append(map)

# find the value of k that has the largest accuracy
i = int(np.argmax(accuracies))
print("k=%d achieved highest Map of %.2f%% on validation data" % (kVals[i], accuracies[i] * 100))

# Create a plot
# Create a scatter plot
plt.figure()
plt.plot(kVals, accuracies)

# Add labels and title
plt.xlabel("K")
plt.ylabel("MaP")
plt.title("Accuracy vs K")

# Show the plot
plt.savefig("kplot.png")
plt.show()

# Train the classifier
print("[INFO] Training with best K")
knn = KNeighborsClassifier(n_neighbors=kVals[i], metric="minkowski")
knn = knn.fit(dataset, labels)
neighbors = knn.kneighbors(query_data, return_distance=False)

# Get the labels obtained by classifier
predictions = []
for i in range(len(neighbors)):
    neighbors_class = [data_labels[j][1] for j in neighbors[i]]
    predictions.append(neighbors_class)

# Evaluate the model
print("[INFO] Evaluating..")
ground_truth = [x[1] for x in query_labels]
p_1 = mpk(ground_truth, predictions, 1)
p_5 = mpk(ground_truth, predictions, 5)

print('P@1=', p_1)
print('P@5=', p_5)
map = mAP(ground_truth, predictions)
print('mAP=', map)

# Convert ground truth and predictions to binary format
num_classes = len(set(labels))
binary_ground_truth = label_binarize(ground_truth, classes=range(num_classes))
binary_predictions = []

for pred in predictions:
    binary_pred = label_binarize(pred, classes=range(num_classes))
    binary_predictions.append(binary_pred.mean(axis=0))

binary_predictions = np.array(binary_predictions)

# Calculate precision-recall curve
precision, recall, _ = precision_recall_curve(binary_ground_truth.ravel(), binary_predictions.ravel())

# Plot precision-recall curve
plt.figure()
plt.plot(recall, precision, marker='.', label='KNN Classifier')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='best')
plt.grid()
plt.savefig("roc.png")
