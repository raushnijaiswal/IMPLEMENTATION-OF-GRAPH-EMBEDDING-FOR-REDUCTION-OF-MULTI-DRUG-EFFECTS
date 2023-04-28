#classifiers in this code do not take graph edges as input. Instead, they take a set of features extracted from the graph as input.
# Import necessary libraries
import pandas as pd
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, RidgeClassifier,SGDClassifier, Perceptron, PassiveAggressiveClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import matplotlib.pyplot as plt

# Load the drug side effects dataset
df = pd.read_csv('/Batch 08 Project File/drug drug side-effect dataset.csv')

#Create a new graph that includes all nodes from the dataset
nodes = set(df['Drug 1']).union(set(df['Drug 2']))
G = nx.Graph()
G.add_nodes_from(nodes)

# Choose a layout algorithm
pos = nx.kamada_kawai_layout(G)

# Set the plot size and label visibility
plt.figure(figsize=(20, 20))

# Choose node colors
node_colors = [G.degree(n) for n in G.nodes()]

# Draw nodes and edges
nx.draw_networkx_nodes(G, pos, node_size=1000, cmap=plt.cm.Blues, node_color='violet')
nx.draw_networkx_edges(G, pos, edge_color='black', alpha=0.5)

# Add the side effect labels to the edges
edge_labels = nx.get_edge_attributes(G, 'label')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_color='red')

# Show the plot
plt.axis('off')
plt.show()


# Split the nodes into training and test sets
X_train_nodes, X_test_nodes = train_test_split(list(G.nodes), test_size=0.2, random_state=42)
# Extract features from the graph
X = []
max_degree = max([G.degree(node) for node in G.nodes])
for node in G.nodes:
    neighbors = set(G.neighbors(node))
    features = [int(node in neighbors)]
    for neighbor in neighbors:
        neighbor_features = [int(neighbor in neighbors)]
        features += neighbor_features
    while len(features) < max_degree + 1:
        features.append(0)
    X.append(features)

# Create a dictionary to map nodes to their index in the feature matrix
node_to_index = {node: i for i, node in enumerate(G.nodes)}

# Create a new target variable with one entry for each drug pair
drug_pairs = [(drug1, drug2) for drug1 in df['Drug 1'].unique() for drug2 in df['Drug 2'].unique() if drug1 != drug2]
y = [int((drug1, drug2) in zip(df['Drug 1'], df['Drug 2'])) for (drug1, drug2) in drug_pairs]

# Filter out drug pairs that are not included in the graph
valid_pairs = [(drug1, drug2) for (drug1, drug2) in drug_pairs if drug1 in node_to_index and drug2 in node_to_index]
X = [X[node_to_index[drug1]] + X[node_to_index[drug2]] for (drug1, drug2) in valid_pairs]
y = [y[i] for i in range(len(y)) if (drug_pairs[i][0], drug_pairs[i][1]) in valid_pairs]

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the length of X_train, y_train, X_test, and y_test
print('Length of X_train:', len(X_train))
print('Length of y_train:', len(y_train))
print('Length of X_test:', len(X_test))
print('Length of y_test:', len(y_test))

# Train and evaluate each classifier on the test data
classifiers = {
    'Logistic Regression': LogisticRegression(),
    'Ridge Classifier': RidgeClassifier(),
    'Support Vector Machine': SVC(kernel='linear'),
    'Random Forest Classifier': RandomForestClassifier(),
    'Perceptron': Perceptron(),
    'Passive Aggressive Classifier': PassiveAggressiveClassifier(),
    'SGD Classifier': SGDClassifier(),
}

for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    confusion = confusion_matrix(y_test, y_pred)

    print(f'{name}:')
    print('Accuracy:', accuracy)
    print('F1 score:', f1)
    print('Confusion matrix:\n', confusion)
    print()

