# zad. 2 - Trebusz (Warwolf)

import random
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

v0 = 50
h = 100
g = 9.81

cel = random.randint(50, 340)
print(f"Cel znajduje się w odległości {cel} metrów.")

trafiony = False
ostatni_kat = None
liczba_prob = 0

while not trafiony:
    kat_stopnie = float(input("Podaj kąt strzału w (w stopniach): "))
    kat = math.radians(kat_stopnie)
    liczba_prob += 1

    dystans = (v0 * math.sin(kat) + math.sqrt(v0**2 * math.sin(kat)**2 + 2 * g * h)) * (v0 * math.cos(kat)) / g

    print(f"Pocisk upadł w odległości {dystans:.2f} m.")

    if cel - 5 <= dystans <= cel + 5:
        print(f"Cel trafiony! Liczba prób: {liczba_prob}")
        trafiony = True
        ostatni_kat = kat
    else:
        print("Chybiony! Spróbuj ponownie.")

# 3) Rysowanie trajektorii trafionego strzału
vx = v0 * math.cos(ostatni_kat)
vy = v0 * math.sin(ostatni_kat)

t_max = (vy + math.sqrt(vy**2 + 2 * g * h)) / g
t = np.linspace(0, t_max, 1000)

x = vx * t
y = vy * t - 0.5 * g * t**2 + h

plt.figure(figsize=(10, 6))
plt.plot(x, y, color='blue', label="Trajektoria pocisku")
plt.axhline(y=0, color='black', linewidth=0.5)
plt.xlabel("Odległość (m)")
plt.ylabel("Wysokość (m)")
plt.title("Trajektoria pocisku Warwolfa")
plt.legend(loc="upper right")
plt.grid(True)
plt.savefig("trajektoria.png")
print("Trajektoria zapisana do trajektoria.png")
