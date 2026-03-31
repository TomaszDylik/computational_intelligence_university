import json
from pathlib import Path

import cv2
from ultralytics import YOLO

# folder z miniaturami ptakow
IMAGES_DIR = "bird_miniatures"

# gdzie zapisujemy raport i obrazy z proby
OUTPUT_DIR = "outputs_ex3_yolo"

# model YOLO użyty w zadaniu
MODEL_FILE = "yolov8n.pt"

# klasy, ktore warto sprawdzac dla latajacych obiektow
TARGET_CLASS_NAMES = ["bird", "airplane", "kite"]

# jedna prosta proba: niski confidence i powiekszenie obrazu
CONFIDENCE_THRESHOLD = 0.01
SCALE_FACTOR = 4


# zapis danych do jsona
def save_json(path, data):
	with open(path, "w", encoding="utf-8") as file:
		json.dump(data, file, indent=2, ensure_ascii=False)


# pobranie listy obrazow z folderu
def get_image_paths(images_dir):
	return sorted(path for path in images_dir.glob("*.jpg"))


# zamiana nazw klas na odpowiadajace im identyfikatory z modelu
def get_target_class_ids(model):
	class_ids = []
	for class_id, class_name in model.names.items():
		if class_name in TARGET_CLASS_NAMES:
			class_ids.append(class_id)
	return class_ids


# przygotowanie obrazu dla YOLO: tylko powiekszenie, zeby ptaki byly troche lepiej widoczne
def preprocess_image(image):
	return cv2.resize(
		image,
		None,
		fx=SCALE_FACTOR,
		fy=SCALE_FACTOR,
		interpolation=cv2.INTER_CUBIC,
	)


# zamiana detekcji YOLO na prosty slownik do raportu
def box_to_dict(box, names):
	class_id = int(box.cls[0])
	return {
		"class_id": class_id,
		"class_name": names[class_id],
		"confidence": round(float(box.conf[0]), 6),
		"bbox_xyxy": [round(float(value), 4) for value in box.xyxy[0].tolist()],
	}


# jedna proba YOLO na pojedynczym obrazie
def detect_on_image(model, image_path, class_ids):
	image = cv2.imread(str(image_path))
	if image is None:
		raise FileNotFoundError(f"Nie mozna wczytac obrazu: {image_path}")

	processed = preprocess_image(image)
	result = model.predict(
		processed,
		conf=CONFIDENCE_THRESHOLD,
		classes=class_ids,
		imgsz=1280,
		verbose=False,
	)[0]

	detections = [box_to_dict(box, model.names) for box in result.boxes]
	return result, detections


# zapis obrazow z bboxami dla tej proby
def save_visualizations(model, image_paths, class_ids, output_dir):
	visual_dir = output_dir / "boxed_images"
	visual_dir.mkdir(exist_ok=True)
	per_image = []

	for image_path in image_paths:
		result, detections = detect_on_image(model, image_path, class_ids)
		output_path = visual_dir / f"{image_path.stem}_boxed.jpg"
		cv2.imwrite(str(output_path), result.plot())

		per_image.append(
			{
				"image": image_path.name,
				"count": len(detections),
				"detections": detections,
			}
		)

	return per_image


# prosty wniosek do raportu i do powiedzenia prowadzacemu
def build_conclusion(total_detections, images_with_detections):
	if total_detections == 0:
		return "YOLO nie wykryl zadnych obiektow. Proba pokazuje, ze ten model w tej konfiguracji nie radzi sobie z tak malymi miniaturami."
	if total_detections <= 2:
		return "YOLO wykryl tylko pojedyncze obiekty. To oznacza, ze zadanie zostalo sprawdzone czesciowo, ale model slabo radzi sobie z bardzo malymi ptakami."
	return "YOLO wykryl czesc obiektow, ale wynik nadal jest niepelny i trzeba traktowac go jako probe, a nie dokladny licznik."


def main():
	base_dir = Path(__file__).resolve().parent
	images_dir = base_dir / IMAGES_DIR
	output_dir = base_dir / OUTPUT_DIR
	output_dir.mkdir(exist_ok=True)

	image_paths = get_image_paths(images_dir)
	if not image_paths:
		print("Brak obrazow do przetworzenia.")
		return

	model = YOLO(str(base_dir / MODEL_FILE))
	class_ids = get_target_class_ids(model)
	per_image = save_visualizations(model, image_paths, class_ids, output_dir)
	total_detections = sum(item["count"] for item in per_image)
	images_with_detections = sum(1 for item in per_image if item["count"] > 0)
	conclusion = build_conclusion(total_detections, images_with_detections)

	report = {
		"task": "YOLO i licznik ptakow - wersja skrocona",
		"model": MODEL_FILE,
		"checked_classes": TARGET_CLASS_NAMES,
		"attempt": {
			"confidence_threshold": CONFIDENCE_THRESHOLD,
			"scale_factor": SCALE_FACTOR,
			"preprocessing": "none",
		},
		"total_detections": total_detections,
		"images_with_detections": images_with_detections,
		"per_image": per_image,
		"conclusion": conclusion,
	}
	save_json(output_dir / "report.json", report)

	print("Proba YOLO:")
	print(f"scale={SCALE_FACTOR}, confidence={CONFIDENCE_THRESHOLD}, preprocessing=none")
	print(
		f"lacznie detekcji: {total_detections}, "
		f"obrazy z detekcjami: {images_with_detections}"
	)
	print()
	print("Liczba obiektow wykrytych przez YOLO na obrazach:")
	for item in per_image:
		print(f"{item['image']}: {item['count']}")

	print()
	print(f"Wniosek: {conclusion}")


if __name__ == "__main__":
	main()
