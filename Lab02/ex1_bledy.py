import pandas as pd
import os

csv_path = os.path.join(os.path.dirname(__file__), 'iris_big_with_errors.csv')
df = pd.read_csv(csv_path, on_bad_lines='skip')

print("\n--- NAPRAWA DANYCH NUMERYCZNYCH ---")

kolumny_liczbowe = df.select_dtypes(include=['float64', 'int64']).columns

for kolumna in kolumny_liczbowe:
    poprawne_dane = df[(df[kolumna] > 0) & (df[kolumna] < 15)][kolumna]
    mediana = poprawne_dane.median()
    
    df.loc[(df[kolumna] <= 0) | (df[kolumna] >= 15), kolumna] = float('nan')
    
    df[kolumna] = df[kolumna].fillna(mediana)

print("Liczby naprawione! Możesz użyć df.describe() by sprawdzić, czy min i max są teraz w normie.")

print("\n--- NAPRAWA GATUNKÓW ---")

nazwa_kolumny = 'target_name' 

print("Przed naprawą:", df[nazwa_kolumny].unique())

df[nazwa_kolumny] = df[nazwa_kolumny].str.lower().str.strip()

poprawki = {
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

    #
}

df[nazwa_kolumny] = df[nazwa_kolumny].replace(poprawki)

print("Po naprawie:", df[nazwa_kolumny].unique())