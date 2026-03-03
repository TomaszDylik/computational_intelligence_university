# zad. 1 - Biorytmy (wersja po korekcie AI - punkt d)

import math
import datetime

# a) Pytanie o imię i datę urodzenia, powitanie, obliczenie biorytmów

imie = input("Podaj swoje imię: ")
print(f"\nWitaj, {imie}! Sprawdź swój biorytm! Jak się czujesz dzisiaj?")

rok = int(input("W jakim roku się urodziłeś/aś? "))
miesiac = int(input("Podaj miesiąc urodzenia (1-12): "))
dzien = int(input("Podaj dzień urodzenia (1-31): "))

dzisiaj = datetime.date.today()
data_urodzenia = datetime.date(rok, miesiac, dzien)
ile_dni = (dzisiaj - data_urodzenia).days

print(f"\n{imie}, dzisiaj jest {ile_dni}. dzień Twojego życia!")


def oblicz_biorytmy(t):
    """Oblicza wartości biorytmów na podstawie liczby dni życia."""
    fizyczny = math.sin(2 * math.pi * t / 23)
    emocjonalny = math.sin(2 * math.pi * t / 28)
    intelektualny = math.sin(2 * math.pi * t / 33)
    return fizyczny, emocjonalny, intelektualny


# b) Ocena biorytmów: wysokie (>0.5), niskie (<-0.5), sprawdzenie jutra

def ocen_biorytmy(fizyczny, emocjonalny, intelektualny, ile_dni):
    """Ocenia wyniki biorytmów i wyświetla odpowiedni komentarz."""
    nazwy = ["fizyczny", "emocjonalny", "intelektualny"]
    wartosci = [fizyczny, emocjonalny, intelektualny]
    jutro = oblicz_biorytmy(ile_dni + 1)

    print(f"\nTwoje biorytmy na dziś:")
    print(f"  Fizyczny:       {fizyczny:.4f}")
    print(f"  Emocjonalny:    {emocjonalny:.4f}")
    print(f"  Intelektualny:  {intelektualny:.4f}\n")

    for i, (nazwa, wartosc) in enumerate(zip(nazwy, wartosci)):
        if wartosc > 0.5:
            print(f"Gratulacje! Twój wynik {nazwa} jest wysoki ({wartosc:.2f}).")
        elif wartosc < -0.5:
            if jutro[i] > wartosc:
                print(f"Twój wynik {nazwa} jest niski ({wartosc:.2f}), ale nie martw się – jutro będzie lepiej!")
            else:
                print(f"Twój wynik {nazwa} jest niski ({wartosc:.2f}).")
        else:
            print(f"Twój wynik {nazwa} jest w normie ({wartosc:.2f}).")


biorytmy = oblicz_biorytmy(ile_dni)
ocen_biorytmy(*biorytmy, ile_dni)

# c) Czas spędzony na zadaniu a-b: ok. 5 minut