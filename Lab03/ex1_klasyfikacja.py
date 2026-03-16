from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

def load_iris_dataframe() -> pd.DataFrame:
	base_dir = Path(__file__).resolve().parent
	data_path = base_dir / "iris_big.csv"
	return pd.read_csv(data_path)

def classify_iris(sl: float, sw: float, pl: float, pw: float) -> str:
	if pl < 3:
		return "setosa"
	elif pw > 2:
		return "virginica"
	else:
		return "versicolor"

def main() -> None:
	df = load_iris_dataframe()

	train_set, test_set = train_test_split(df.values, train_size=0.7, random_state=298655)

	good_predictions = 0

	for row in test_set:
		if classify_iris(row[0], row[1], row[2], row[3]) == row[4]:
			good_predictions += 1

	accuracy = good_predictions / len(test_set)

	print(f"Liczba poprawnych predykcji: {good_predictions}/{len(test_set)}")
	print(f"Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
	main()
