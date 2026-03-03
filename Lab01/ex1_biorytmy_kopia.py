# zad. 1 - Biorytmy (kopia oryginału przed korektą AI - punkt d)

import math
import datetime

# a)
imie = input("Podaj swoje imię: ")
print("Witaj, " + imie + "!")

print("W jakim roku się urodziłeś?")
rok = int(input())
print("Podaj miesiąc swojego urodzenia")
miesiac = int(input())
print("Podaj dzień swojego urodzenia")
dzien = int(input())

dzisiaj = datetime.date.today()
data_urodzenia = datetime.date(rok, miesiac, dzien)
wiek = (dzisiaj - data_urodzenia)
ile_dni = wiek.days

print("Dzisiaj jest " + str(ile_dni) + ". dzień Twojego życia!")

def oblicz_biorytmy(ile_dni): 
    fizyczny = math.sin(2 * math.pi * ile_dni / 23)
    emocjonalny = math.sin(2 * math.pi * ile_dni / 28)
    intelektualny = math.sin(2 * math.pi * ile_dni / 33)
    return fizyczny, emocjonalny, intelektualny

def kolejny_dzien(ile_dni):
    fizyczny = math.sin(2 * math.pi * (ile_dni + 1) / 23)
    emocjonalny = math.sin(2 * math.pi * (ile_dni + 1) / 28)
    intelektualny = math.sin(2 * math.pi * (ile_dni + 1) / 33)
    return fizyczny, emocjonalny, intelektualny

# b)
def czy_wysoki(fizyczny, emocjonalny, intelektualny):
    print(fizyczny, emocjonalny, intelektualny)
    kolejny = kolejny_dzien(ile_dni)
    if fizyczny > 0.5:
        print("Gratuluje wysokiego wyniku fizycznego.")
    if fizyczny < -0.5:
        if kolejny[0] > fizyczny:
            print("Nie przejmuj się twój wynik fizyczny jest niski - jutro bedzie lepiej.")
        else:
            print("Nie przejmuj się twój wynik fizyczny jest niski")

    if emocjonalny > 0.5:
        print("Gratuluje wysokiego wyniku emocjonalnego.")
    if emocjonalny < -0.5:
        if kolejny[1] > emocjonalny:
            print("Twój wynik emocjonalny jest niski, ale jutro bedzie lepiej.")
        else:
            print("Twój wynik emocjonalny jest niski")

    if intelektualny > 0.5:
        print("Gratuluje wysokiego wyniku intelektualnego.")
    if intelektualny < -0.5:
        if kolejny[2] > intelektualny:
            print("Twój wynik intelektualny jest niski, ale jutro bedzie lepiej.")
        else:
            print("Nie przejmuj się twój wynik intelektualny jest niski")

czy_wysoki(*oblicz_biorytmy(ile_dni))

# c) Czas spędzony na zadaniu a-b: ok. 30 minut