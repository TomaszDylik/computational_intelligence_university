import json
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO

# progi
THRESHOLDS = [0.1, 0.3, 0.5, 0.7]

# pliki
IMAGE_FILE = "office_yolo.png"
VIDEO_FILES = ["office_yolo.mp4", "street_yolo.mp4"]

# zapisywanie jsona
def save_json(path, data):
	with open(path, "w", encoding="utf-8") as file:
		json.dump(data, file, indent=2, ensure_ascii=False)

# zamiana yolo na normalne boxy
def box_to_dict(box, names):
	class_id = int(box.cls[0])
	return {
		"class_id": class_id,
		"class_name": names[class_id],
		"confidence": round(float(box.conf[0]), 6),
		"bbox_xyxy": [round(float(value), 4) for value in box.xyxy[0].tolist()],
	}

# zapis informacji o modelu do jsona
def save_model_info(model, output_dir):
	info = {
		"model": "yolov8n.pt",
		"dataset": "COCO",
		"number_of_classes": len(model.names),
		"cnn_inside": "YOLOv8n uses a convolutional neural network backbone with C2f blocks and a detection head.",
	}
	save_json(output_dir / "model_info.json", info)


def detect_on_image(model, image_path, confidence, output_dir):
	result = model.predict(str(image_path), conf=confidence, verbose=False)[0]

	detections = [box_to_dict(box, model.names) for box in result.boxes]
	
	# zapis wyników detekcji do jsona
	save_json(
		output_dir / f"image_{image_path.stem}_conf_{confidence:.1f}.json",
		{
			"source": image_path.name,
			"confidence_threshold": confidence,
			"detections": detections,
		},
	)

	# zapis obrazu z boxami
	cv2.imwrite(
		str(output_dir / f"image_{image_path.stem}_conf_{confidence:.1f}_boxed.png"),
		result.plot(),
	)


def detect_on_video(model, video_path, confidence, output_dir):
	# otwieranie filmu do odczytu
	video = cv2.VideoCapture(str(video_path))
	if not video.isOpened():
		raise FileNotFoundError(f"Nie mozna otworzyc pliku wideo: {video_path}")

	# odczyt podstawowych informacji o filmie
	fps = video.get(cv2.CAP_PROP_FPS)
	if fps <= 0:
		fps = 25.0

	width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

	# nowy wideo do zapisu wyników detekcji
	writer = cv2.VideoWriter(
		str(output_dir / f"video_{video_path.stem}_conf_{confidence:.1f}_boxed.mp4"),
		cv2.VideoWriter_fourcc(*"mp4v"),
		fps,
		(width, height),
	)

	# wyniki dla kazdej klatki
	all_frames = []
	# zlicza wystapienia danje klasy
	class_counts = Counter()
	frame_number = 0

	while True:
		success, frame = video.read()
		if not success:
			break

		# detekcja w klatce
		result = model.predict(frame, conf=confidence, verbose=False)[0]
		detections = [box_to_dict(box, model.names) for box in result.boxes]

		for detection in detections:
			class_counts[detection["class_name"]] += 1

		# zapsiywanie jsona z filmu
		all_frames.append(
			{
				"frame_number": frame_number,
				"time_sec": round(frame_number / fps, 4),
				"detections": detections,
			}
		)

		# Ta klatka zostaje zapisana do filmu wyjściowego już z narysowanymi bboxami.
		writer.write(result.plot())
		frame_number += 1

	# zamkniecie plikow
	video.release()
	writer.release()

	# zapis końcowego jsona
	save_json(
		output_dir / f"video_{video_path.stem}_conf_{confidence:.1f}.json",
		{
			"source": video_path.name,
			"confidence_threshold": confidence,
			"class_counts": dict(class_counts),
			"frames": all_frames,
		},
	)


def main():
	base_dir = Path(__file__).resolve().parent
	output_dir = base_dir / "outputs"
	output_dir.mkdir(exist_ok=True)

	# wczytanie modelu
	model = YOLO(str(base_dir / "yolov8n.pt"))
	save_model_info(model, output_dir)

	image_path = base_dir / IMAGE_FILE
	video_paths = [base_dir / name for name in VIDEO_FILES]

	# detekcja dla kazdego confidence
	for confidence in THRESHOLDS:
		detect_on_image(model, image_path, confidence, output_dir)
		for video_path in video_paths:
			detect_on_video(model, video_path, confidence, output_dir)



if __name__ == "__main__":
	main()

