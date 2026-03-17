import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from pathlib import Path

data_path = Path(__file__).resolve().parent / "iris_big.csv"
df = pd.read_csv(data_path)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# a) Podział 70/30
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=298655
)

classifiers = {
    "k-NN k=3": KNeighborsClassifier(n_neighbors=3),
    "k-NN k=5": KNeighborsClassifier(n_neighbors=5),
    "k-NN k=11": KNeighborsClassifier(n_neighbors=11),
    "Naive Bayes": GaussianNB(),
    "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=298655),
    "DecTree (zad2)": DecisionTreeClassifier(random_state=298655),
}

results = {}
classes = sorted(y.unique())

for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred) * 100
    results[name] = acc

    cm = confusion_matrix(y_test, y_pred, labels=classes)

    print(f"\n{name}")
    print(f"Dokładność: {acc:.2f}%")
    print("Macierz błędów:")
    print(cm)

print("\nPORÓWNANIE")
sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
for rank, (name, acc) in enumerate(sorted_results, start=1):
    print(f"{rank}. {name}: {acc:.2f}%")


