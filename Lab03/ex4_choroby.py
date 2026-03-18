import matplotlib
import pandas as pd
import seaborn as sns
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 298655
TEST_SIZE = 0.25
DATA_FILE = r"C:\Users\tomek\Desktop\computational_intelligence_university\Lab03\diagnosis.csv"
OUT_3D_PNG = r"C:\Users\tomek\Desktop\computational_intelligence_university\Lab03\ex4_wykres_3d.png"
OUT_3D_GIF = r"C:\Users\tomek\Desktop\computational_intelligence_university\Lab03\ex4_wykres_3d_rotacja.gif"
OUT_METRICS = r"C:\Users\tomek\Desktop\computational_intelligence_university\Lab03\ex4_tabela_miar.png"
OUT_CONFUSION = r"C:\Users\tomek\Desktop\computational_intelligence_university\Lab03\ex4_macierze_bledow.png"

## podpuntk a)
def plot_3d_points(df: pd.DataFrame, output_path: str) -> None:
	fig = plt.figure(figsize=(10, 8))
	ax = fig.add_subplot(111, projection="3d")

	healthy = df[df["diagnosis"] == 0]
	sick = df[df["diagnosis"] == 1]

	ax.scatter(
		healthy["param1"],
		healthy["param2"],
		healthy["param3"],
		c="royalblue",
		alpha=0.7,
		label="Zdrowy (0)",
	)
	ax.scatter(
		sick["param1"],
		sick["param2"],
		sick["param3"],
		c="crimson",
		alpha=0.7,
		label="Chory (1)",
	)

	ax.set_title("Wykres 3D danych medycznych")
	ax.set_xlabel("param1")
	ax.set_ylabel("param2")
	ax.set_zlabel("param3")
	ax.legend(loc="upper right")
	plt.tight_layout()
	
	plt.savefig(output_path, dpi=150)
	plt.close(fig)

## gif z rotującym wykresem 3D
def plot_3d_rotating_gif(df: pd.DataFrame, output_path: str) -> None:
	fig = plt.figure(figsize=(10, 8))
	ax = fig.add_subplot(111, projection="3d")

	healthy = df[df["diagnosis"] == 0]
	sick = df[df["diagnosis"] == 1]

	ax.scatter(
		healthy["param1"],
		healthy["param2"],
		healthy["param3"],
		c="royalblue",
		alpha=0.7,
		label="Zdrowy (0)",
	)
	ax.scatter(
		sick["param1"],
		sick["param2"],
		sick["param3"],
		c="crimson",
		alpha=0.7,
		label="Chory (1)",
	)

	ax.set_title("Wykres 3D danych medycznych (rotacja)")
	ax.set_xlabel("param1")
	ax.set_ylabel("param2")
	ax.set_zlabel("param3")
	ax.legend(loc="upper right")

	def update(angle):
		ax.view_init(elev=24, azim=angle)
		return ax

	ani = FuncAnimation(fig, update, frames=range(0, 360, 5), interval=80, blit=False)

	try:
		ani.save(output_path, writer=PillowWriter(fps=12), dpi=120)
	except Exception as exc:
		print(f"Nie udalo sie zapisac GIF: {exc}")
	finally:
		plt.close(fig)


def evaluate_classifiers(X_train, X_test, y_train, y_test):
	models = {
		"Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
		"KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
		"Gaussian NB": GaussianNB(),
		"MLP": MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=2000, random_state=RANDOM_STATE),
	}

	metrics_rows = []
	confusion_matrices = {}

	for name, model in models.items():
		model.fit(X_train, y_train)
		y_pred = model.predict(X_test)

		metrics_rows.append(
			{
				"Klasyfikator": name,
				"Accuracy": accuracy_score(y_test, y_pred),
				"Precision": precision_score(y_test, y_pred, zero_division=0),
				"Recall/Sensitivity": recall_score(y_test, y_pred, zero_division=0),
			}
		)
		
		confusion_matrices[name] = confusion_matrix(y_test, y_pred)

	metrics_df = pd.DataFrame(metrics_rows).set_index("Klasyfikator")
	return metrics_df, confusion_matrices

## podpuntk b - heatmapa
def plot_metrics_table(metrics_df: pd.DataFrame, output_path: str) -> None:
	plt.figure(figsize=(9, 4))
	sns.heatmap(
		metrics_df,
		annot=True,
		fmt=".3f",
		cmap="YlGnBu",
		linewidths=0.5,
		cbar=True,
		vmin=0.7,
		vmax=1,
	)
	plt.title("Porownanie miar klasyfikatorow")
	plt.xlabel("Miary")
	plt.ylabel("Klasyfikator")
	plt.tight_layout()
	plt.savefig(output_path, dpi=150)
	plt.close()

## podpunkt b - macierz błędów
def plot_confusion_matrices(conf_mats: dict, output_path: str) -> None:
	fig, axes = plt.subplots(2, 2, figsize=(10, 8))
	axes = axes.flatten()

	for ax, (name, cm) in zip(axes, conf_mats.items()):
		sns.heatmap(
			cm,
			annot=True,
			fmt="d",
			cmap="Blues",
			cbar=False,
			ax=ax,
			xticklabels=["Pred 0", "Pred 1"],
			yticklabels=["True 0", "True 1"],
		)
		ax.set_title(name)
		ax.set_xlabel("Predykcja")
		ax.set_ylabel("Rzeczywista")

	fig.suptitle("Macierze bledow (confusion matrix)", fontsize=14)
	plt.tight_layout()
	plt.savefig(output_path, dpi=150)
	plt.close(fig)


"""
podpunkt c i d - wnioski

1. Accuracy: 
   - Jaki procent wszystkich decyzji modelu był trafny.
   - Jest  myląca przy niezbalansowanych danych. Jeśli masz 99 zdrowych i 1 chorego, 
     model mówiący ZAWSZE "zdrowy" osiągnie 99% Accuracy, choć jest całkowicie bezużyteczny.

2. Precision: 
   - Z grupy ludzi, których model uznał za "chorych", ilu jest naprawdę chorych.
   - Gdy chcemy uniknąć fałszywych alarmów. 

3. Recall / Sensitivity: 
   - Jaki procent wszytskich chorych pacjentów nasz model poprawnie wyłapał?
   - Gdy przegapienie chorego to tragedia (minimalizujemy False Negatives - FN). 
     Np. przy badaniach przesiewowych – lepiej wszcząć fałszywy alarm, niż odesłać chorego do domu.

Wniosek: 
Przy nierównych klasach nigdy nie ufaj samemu Accuracy. Zawsze patrz na Precision, Recall 
oraz macierz błędów (Confusion Matrix), żeby dokładnie widzieć, w którą stronę model się myli.
"""

def main() -> None:
	df = pd.read_csv(DATA_FILE)
	X = df[["param1", "param2", "param3"]]
	y = df["diagnosis"]

	plot_3d_points(df, OUT_3D_PNG)
	plot_3d_rotating_gif(df, OUT_3D_GIF)

	X_train, X_test, y_train, y_test = train_test_split(
		X,
		y,
		test_size=TEST_SIZE,
		random_state=RANDOM_STATE,
		stratify=y,
	)

	metrics_df, conf_mats = evaluate_classifiers(X_train, X_test, y_train, y_test)

	print("Porownanie klasyfikatorow (punkt b):")
	print(metrics_df.round(3))

	plot_metrics_table(metrics_df, OUT_METRICS)
	plot_confusion_matrices(conf_mats, OUT_CONFUSION)

if __name__ == "__main__":
	main()

