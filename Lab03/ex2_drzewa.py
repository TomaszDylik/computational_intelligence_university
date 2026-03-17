import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from pathlib import Path

data_path = Path(__file__).resolve().parent / "iris_big.csv"
df = pd.read_csv(data_path)

X = df.iloc[:, :-1] 
y = df.iloc[:, -1]  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=298655)


print("--- Zbiór treningowy (Inputy) ---")
print(X_train.head())
print("\n--- Zbiór testowy (Inputy) ---")
print(X_test.head())

clf = DecisionTreeClassifier(random_state=298655)
clf.fit(X_train, y_train)

plt.figure(figsize=(20, 12))
plot_tree(clf, feature_names=X.columns, class_names=clf.classes_, 
          filled=True, rounded=True, fontsize=10)
plt.title("Automatycznie wygenerowane Drzewo Decyzyjne Iris", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(Path(__file__).resolve().parent / "drzewo_decyzyjne.png", dpi=100, bbox_inches='tight')

accuracy = clf.score(X_test, y_test)
print(f"\nDokładność (Accuracy) na zbiorze testowym: {accuracy * 100:.2f}%")

y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)

plt.figure(figsize=(8, 6))
disp.plot(cmap=plt.cm.Blues)
plt.title("Macierz Błędów (Confusion Matrix)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(Path(__file__).resolve().parent / "confusion_matrix.png", dpi=100, bbox_inches='tight')

print(f"Liczba próbek w zbiorze testowym: {len(y_test)}")
print(f"Liczba poprawnych predykcji: {(y_pred == y_test).sum()}")
print(f"Liczba błędnych predykcji: {(y_pred != y_test).sum()}")
print(f"\nGatunki Iris w zbiorze: {list(clf.classes_)}")