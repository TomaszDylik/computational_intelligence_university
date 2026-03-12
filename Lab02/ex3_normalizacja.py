import os

import matplotlib
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt

base_dir = os.path.dirname(__file__)
csv_path = os.path.join(base_dir, 'iris_big.csv')
plot_path = os.path.join(base_dir, 'iris_normalizacja.png')

df = pd.read_csv(csv_path)
x_column = 'sepal length (cm)'
y_column = 'sepal width (cm)'
species_column = 'target_name'

original = df[[x_column, y_column, species_column]].copy()

# Normalizacja min-max.
min_max = original.copy()
for column in [x_column, y_column]:
	minimum = original[column].min()
	maximum = original[column].max()
	min_max[column] = (original[column] - minimum) / (maximum - minimum)

# Skalowanie z-score.
z_score = original.copy()
for column in [x_column, y_column]:
	mean = original[column].mean()
	std = original[column].std(ddof=0)
	z_score[column] = (original[column] - mean) / std


def print_stats(name, data):
	print(f'\n{name}:')
	stats = data[[x_column, y_column]].agg(['min', 'max', 'mean', 'std'])
	print(stats.to_string())


print_stats('dane_oryginalne', original)
print_stats('dane_min_max', min_max)
print_stats('dane_z_score', z_score)

print('\nwnioski:')
print('min-max ustawia minimum na 0 i maksimum na 1.')
print('z-score daje srednia bliska 0 i odchylenie standardowe bliskie 1.')
print('normalizacja nie zmienia klas irysow, tylko skale osi.')

colors = {
	'setosa': 'tab:blue',
	'versicolor': 'tab:orange',
	'virginica': 'tab:green',
}

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

datasets = [
	('Dane oryginalne', original),
	('Min-max', min_max),
	('Z-score', z_score),
]

for axis, (title, data) in zip(axes, datasets):
	for species in data[species_column].unique():
		mask = data[species_column] == species
		axis.scatter(
			data.loc[mask, x_column],
			data.loc[mask, y_column],
			label=species,
			color=colors.get(species, 'gray'),
			alpha=0.7,
		)

	axis.set_title(title)
	axis.set_xlabel(x_column)
	axis.set_ylabel(y_column)
	axis.legend()

plt.tight_layout()
plt.savefig(plot_path, dpi=150)
plt.close()

print('\nzapisano_wykres:')
print(plot_path)
