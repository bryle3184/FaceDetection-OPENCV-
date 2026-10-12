import argparse
import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-m", "--model", default="onnx/models/yunet-202303.onnx", help="path to YuNet ONNX model")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="min probability")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
if image is None:
    raise SystemExit(f"[ERROR] Could not load image from '{args['image']}'. Check that the file exists and is a valid image.")
h, w = image.shape[:2]

detector = cv2.FaceDetectorYN.create(
    model=args["model"],
    config="",
    input_size=(w, h),
    score_threshold=args["confidence"],
)

_, faces = detector.detect(image)

if faces is not None:
    for face in faces:
        box = face[0:4].astype("int")
        confidence = face[-1]
        startX, startY, boxW, boxH = box
        endX, endY = startX + boxW, startY + boxH

        text = f"{confidence * 100:.2f}%"
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
        cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

cv2.imshow("Output", image)
cv2.waitKey(0)