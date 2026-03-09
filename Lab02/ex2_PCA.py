import os

import matplotlib
import pandas as pd
from sklearn.decomposition import PCA

matplotlib.use('Agg') # generowanie wykresow bez wyswietlania ich na ekranie
import matplotlib.pyplot as plt


# Przygotowanie sciezek do plikow wejsciowych i wyjsciowych.
base_dir = os.path.dirname(__file__) # sciezka do katalogu gdzie jest skrypt
csv_path = os.path.join(base_dir, 'iris_big.csv')
output_2d_path = os.path.join(base_dir, 'iris_pca_2d.csv')
output_3d_path = os.path.join(base_dir, 'iris_pca_3d.csv')
plot_2d_path = os.path.join(base_dir, 'iris_pca_2d.png')
plot_3d_path = os.path.join(base_dir, 'iris_pca_3d.png')

# Wczytanie calej tabeli z pliku CSV.
df = pd.read_csv(csv_path)

species_column = 'target_name'
feature_columns = []

# Zostawiamy tylko kolumny numeryczne, bo PCA dziala na liczbach.
for column in df.columns:
    if column != species_column:
        feature_columns.append(column)

X = df[feature_columns]
y = df[species_column]

# Najpierw liczymy PCA dla wszystkich 4 skladowych, aby sprawdzic wariancje.
pca_full = PCA(n_components=4)
pca_full.fit(X)

explained_variance = pca_full.explained_variance_ratio_
variance_2d = explained_variance[:2].sum()
variance_3d = explained_variance[:3].sum()
loss_2d = 1 - variance_2d
loss_3d = 1 - variance_3d

# Wypisanie, ile informacji daje kazda skladowa i ile zostaje po redukcji.
print('wariancja_wyjasniona_przez_kolejne_skladowe:')
for index, value in enumerate(explained_variance, start=1):
	print(f'PC{index}: {value:.6f}')

print('\nsuma_wariancji:')
print(f'2 skladowe: {variance_2d:.6f}')
print(f'3 skladowe: {variance_3d:.6f}')

print('\nstrata_informacji:')
print(f'po_usunieciu_2_kolumn: {loss_2d:.6f}')
print(f'po_usunieciu_1_kolumny: {loss_3d:.6f}')

print('\nwniosek: mozna usunac 2 kolumny, bo zostaje ponad 95% wariancji.')

# Tworzymy nowa baze 2D z kolumnami PC1 i PC2.
pca_2d = PCA(n_components=2)
data_2d = pca_2d.fit_transform(X)
df_2d = pd.DataFrame(data_2d, columns=['PC1', 'PC2'])
df_2d[species_column] = y
df_2d.to_csv(output_2d_path, index=False)

# Tworzymy tez wersje 3D z kolumnami PC1, PC2 i PC3.
pca_3d = PCA(n_components=3)
data_3d = pca_3d.fit_transform(X)
df_3d = pd.DataFrame(data_3d, columns=['PC1', 'PC2', 'PC3'])
df_3d[species_column] = y
df_3d.to_csv(output_3d_path, index=False)

# Kolory punktow dla poszczegolnych gatunkow na wykresach.
colors = {
	'setosa': 'tab:blue',
	'versicolor': 'tab:red',
	'virginica': 'tab:green',
}

# Wykres 2D po redukcji do dwoch skladowych.
plt.figure(figsize=(8, 6))
for species in y.unique():
	mask = y == species
	plt.scatter(
		df_2d.loc[mask, 'PC1'],
		df_2d.loc[mask, 'PC2'],
		label=species,
		color=colors.get(species, 'gray'),
		alpha=0.7,
	)

plt.title('PCA irysow - 2 skladowe')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.tight_layout()
plt.savefig(plot_2d_path, dpi=150)
plt.close()

# Wykres 3D po redukcji do trzech skladowych.
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection='3d')

for species in y.unique():
	mask = y == species
	ax.scatter(
		df_3d.loc[mask, 'PC1'],
		df_3d.loc[mask, 'PC2'],
		df_3d.loc[mask, 'PC3'],
		label=species,
		color=colors.get(species, 'gray'),
		alpha=0.7,
	)

ax.set_title('PCA irysow - 3 skladowe')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.legend()
plt.tight_layout()
plt.savefig(plot_3d_path, dpi=150)
plt.close()

# Wypisanie nazw zapisanych plikow wynikowych.
print('\nzapisano:')
print(output_2d_path)
print(output_3d_path)
print(plot_2d_path)
print(plot_3d_path)
