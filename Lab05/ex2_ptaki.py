from pathlib import Path

import cv2
import numpy as np

# ptaki
IMAGES_DIR = "bird_miniatures"

# minimalna i maksymalna wielkosc obiektu, ktory uznajemy za ptaka
MIN_COMPONENT_AREA = 2
MAX_COMPONENT_AREA = 300

# prog po odjeciu tla
THRESHOLD_VALUE = 8

# pobranie wszystkich obrazow z folderu
def get_image_paths(images_dir):
	return sorted(
		path for path in images_dir.iterdir()
	)

# przygotowanie maski, na ktorej ptaki beda bialymi obiektami - ChatGPT
def build_bird_mask(gray_image):
	# lekkie wzmocnienie lokalnego kontrastu, zeby ciemne ptaki byly lepiej widoczne
	enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray_image)

	# rozmyty obraz traktujemy jako przyblizenie tla
	background = cv2.GaussianBlur(enhanced, (0, 0), 9)

	# odejmujemy ciemniejsze obiekty od tla, dzieki czemu ptaki wychodza mocniej
	difference = cv2.subtract(background, enhanced)

	# zamiana na obraz binarny: ptaki beda biale, tlo czarne
	_, mask = cv2.threshold(difference, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)

	# usuniecie szumu z pojedynczych pikseli
	kernel = np.ones((2, 2), np.uint8)
	cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

	return cleaned_mask


# policzenie spojnych bialych plam - ptakow 
def count_birds_from_mask(mask):
	num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
	count = 0

	# etykieta 0 to tlo, dlatego zaczynamy od 1
	for label in range(1, num_labels):
		area = stats[label, cv2.CC_STAT_AREA]
		if MIN_COMPONENT_AREA <= area <= MAX_COMPONENT_AREA:
			count += 1

	return count


# liczenie ptakow na obrazie
def count_birds_in_image(image_path):
	image = cv2.imread(str(image_path))

	# konwersja do skali szarosci zgodnie z poleceniem - ChatGPT
	gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	mask = build_bird_mask(gray_image)
	return count_birds_from_mask(mask)


def main():
	base_dir = Path(__file__).resolve().parent
	images_dir = base_dir / IMAGES_DIR

	image_paths = get_image_paths(images_dir)

	print("Liczba ptakow na obrazach:")
	for image_path in image_paths:
		bird_count = count_birds_in_image(image_path)
		print(f"{image_path.name}: {bird_count}")

if __name__ == "__main__":
	main()
