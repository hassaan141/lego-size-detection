from ultralytics import YOLO
import cv2

model_path = r"weights\best.pt"

model=YOLO(model_path)

# test_path = r"images\720X720-photo-all-bricks.jpg"

# results = model.predict(source=test_path, conf=0.25)
# # print(results)

# image = results[0].plot()
# resize = cv.resize(image, (720,720))
# print(image)

# cv.imshow("Inference Result", resize)
# cv.waitKey(0)
# cv.destroyAllWindows()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Run inference on the captured frame
    results = model.predict(source=frame, conf=0.25)
    
    # Get the annotated frame (this method returns the frame with predictions overlaid)
    annotated_frame = results[0].plot()  
    
    # Display the resulting frame
    cv2.imshow("Live Detection", annotated_frame)

    # Exit loop on pressing 'q'
    if cv2.waitKey(1) == ord("q"):
        break

# When everything is done, release the capture and close windows
cap.release()
cv2.destroyAllWindows()