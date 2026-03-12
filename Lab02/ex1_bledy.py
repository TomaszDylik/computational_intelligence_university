import os
import pandas as pd

base_dir = os.path.dirname(__file__)
csv_path = os.path.join(base_dir, 'iris_big_with_errors.csv')
output_path = os.path.join(base_dir, 'iris_big_cleaned.csv')

df_raw = pd.read_csv(csv_path, on_bad_lines='skip')

species_column = 'target_name'
feature_columns = [column for column in df_raw.columns if column != species_column]

df_dirty = df_raw.copy()

for column in feature_columns:
    df_dirty[column] = pd.to_numeric(df_dirty[column], errors='coerce')

# Dane sa w 5 kolumnach: 4 numeryczne i 1 z gatunkiem.
# Strukture psuja wiersze, ktore maja zla liczbe kolumn.

print('\nbraki_danych:')
print(df_dirty.isna().sum().to_string())
print('suma_brakow:', int(df_dirty.isna().sum().sum()))

df_clean = df_dirty.copy()

for column in feature_columns:
    valid_mask = df_clean[column].between(0, 15, inclusive='neither')
    median = df_clean.loc[valid_mask, column].median()
    df_clean.loc[~valid_mask, column] = pd.NA
    df_clean[column] = df_clean[column].fillna(median)

species = df_clean[species_column].astype('string').str.lower().str.strip()
unique_species_before = sorted(species.dropna().unique().tolist())
species_fixes = {
    'setossa': 'setosa',
    'versicolour': 'versicolor',
    'virginicaa': 'virginica',
    'iris virginica': 'virginica',
    'iris_versicolor': 'versicolor',
    'iris-setosa': 'setosa',
    'virginica?': 'virginica',
    'versicolr': 'versicolor',
    'versi-color': 'versicolor',
    'setosa.': 'setosa',
    'unknown': pd.NA,
}
valid_species = ['setosa', 'versicolor', 'virginica']

species = species.replace(species_fixes)
df_clean[species_column] = species
df_clean.to_csv(output_path, index=False)

print('\nunikatowe_gatunki_przed:', unique_species_before)
print('gatunki_po_naprawie:', sorted(df_clean[species_column].dropna().unique().tolist()))
print('\nzapisano:', output_path)