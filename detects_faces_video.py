# import the necessary packages
from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", default="onnx/models/yunet-202303.onnx",
	help="path to YuNet ONNX model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

detector = None

# loop over the frames from the video stream
while True:
	frame = vs.read()
	if frame is None:
		continue
	frame = imutils.resize(frame, width=400)
	(h, w) = frame.shape[:2]

	if detector is None:
		detector = cv2.FaceDetectorYN.create(
			model=args["model"],
			config="",
			input_size=(w, h),
			score_threshold=args["confidence"],
		)
	else:
		detector.setInputSize((w, h))

	_, faces = detector.detect(frame)

	if faces is not None:
		for face in faces:
			box = face[0:4].astype("int")
			confidence = face[-1]
			startX, startY, boxW, boxH = box
			endX, endY = startX + boxW, startY + boxH

			text = f"{confidence * 100:.2f}%"
			y = startY - 10 if startY - 10 > 10 else startY + 10
			cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
			cv2.putText(frame, text, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	if key == ord("q"):
		break

cv2.destroyAllWindows()
vs.stop()

