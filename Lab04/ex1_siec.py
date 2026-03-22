import math

# Funkcja aktywacji: mapuje dowolne z na zakres (0, 1).
def sigmoid(z: float) -> float:
	return 1.0 / (1.0 + math.exp(-z))


# Blad sredniokwadratowy dla jednej probki.
def mse_loss(y_hat: float, y_true: float) -> float:
	return 0.5 * (y_hat - y_true) ** 2


# Jeden pelny krok uczenia: forward, loss, backprop i aktualizacja wag.
def forward_propagation() -> None:
	x1 = 0.6
	x2 = 0.1
	y_true = 0.8

	eta = 0.1

	w1, w2, b1 = 0.20, -0.30, 0.40
	w3, w4, b2 = -0.50, 0.10, -0.20
	w5, w6, b3 = 0.30, -0.40, 0.20

	print("=== FORWARD PROPAGATION ===")

	# Hidden neuron h1
	z_h1 = x1 * w1 + x2 * w2 + b1
	h1 = sigmoid(z_h1)

	# Hidden neuron h2
	z_h2 = x1 * w3 + x2 * w4 + b2
	h2 = sigmoid(z_h2)

	# Output neuron y_hat (w tym zadaniu bez aktywacji na wyjsciu)
	z_o = h1 * w5 + h2 * w6 + b3
	y_hat = z_o

	print(f"z_h1 = {z_h1:.6f}, h1 = {h1:.6f}")
	print(f"z_h2 = {z_h2:.6f}, h2 = {h2:.6f}")
	print(f"z_o  = {z_o:.6f}, y_hat = {y_hat:.6f} (powinno byc blisko 0.234)")

	loss = mse_loss(y_hat, y_true)
	print(f"Loss = 0.5 * (y_hat - y)^2 = {loss:.6f}")

	print("\n=== BACKPROPAGATION ===")
	delta_out = y_hat - y_true

	# Gradienty warstwy wyjściowej
	dL_dw5 = delta_out * h1
	dL_dw6 = delta_out * h2
	dL_db3 = delta_out

	# Propagacja błędu do warstwy ukrytej
	delta_h1 = delta_out * w5 * h1 * (1.0 - h1)  # dL/dz_h1
	delta_h2 = delta_out * w6 * h2 * (1.0 - h2)  # dL/dz_h2

	# Gradienty dla wag wejście -> warstwa ukryta
	dL_dw1 = delta_h1 * x1
	dL_dw2 = delta_h1 * x2
	dL_db1 = delta_h1

	dL_dw3 = delta_h2 * x1
	dL_dw4 = delta_h2 * x2
	dL_db2 = delta_h2

	print(f"dL/dw1 = {dL_dw1:.6f}")
	print(f"dL/dw2 = {dL_dw2:.6f}")
	print(f"dL/dw3 = {dL_dw3:.6f}")
	print(f"dL/dw4 = {dL_dw4:.6f}")
	print(f"dL/dw5 = {dL_dw5:.6f}")
	print(f"dL/dw6 = {dL_dw6:.6f}")

	print("\n=== AKTUALIZACJA PARAMETROW ===")
	# Gradient descent: theta_new = theta_old - eta * gradient
	w1_new = w1 - eta * dL_dw1
	w2_new = w2 - eta * dL_dw2
	w3_new = w3 - eta * dL_dw3
	w4_new = w4 - eta * dL_dw4
	w5_new = w5 - eta * dL_dw5
	w6_new = w6 - eta * dL_dw6

	b1_new = b1 - eta * dL_db1
	b2_new = b2 - eta * dL_db2
	b3_new = b3 - eta * dL_db3

	print(f"w1: {w1:.6f} -> {w1_new:.6f} (oczekiwane ~0.2024)")
	print(f"w2: {w2:.6f} -> {w2_new:.6f}")
	print(f"w3: {w3:.6f} -> {w3_new:.6f}")
	print(f"w4: {w4:.6f} -> {w4_new:.6f}")
	print(f"w5: {w5:.6f} -> {w5_new:.6f}")
	print(f"w6: {w6:.6f} -> {w6_new:.6f}")
	print(f"b1: {b1:.6f} -> {b1_new:.6f}")
	print(f"b2: {b2:.6f} -> {b2_new:.6f}")
	print(f"b3: {b3:.6f} -> {b3_new:.6f}")

if __name__ == "__main__":
	forward_propagation()
