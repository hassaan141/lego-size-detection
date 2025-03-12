from ultralytics import YOLO
import cv2
# import numpy as np
# from segment_anything import SamPredictor, sam_model_registry
# import torch

model_path = r"weights\best.pt"

model=YOLO(model_path)

test_path = r"images\lego-red-brick.jpg"

results = model.predict(source=test_path, conf=0.25)
# print(results)

# image = results[0].plot()
# resize = cv.resize(image, (720,720))
# boxes = results[0].boxes.xyxy[0].cpu().numpy()
# print(boxes)
# x1,y1,x2,y2=boxes
# cv.circle(image, (round((x1+x2)/2), round((y1+y2)/2)), 5, (255, 0, 255), -1)
# cv.imshow("Inference Result", resize)
# cv.waitKey(0)
# cv.destroyAllWindows()

# segmented_mask = segment_with_sam(test_path, (cx, cy))

# Show segmented mask
# for i, mask in enumerate(segmented_mask):
#     segmented_mask = (mask * 255).astype(np.uint8)
#     cv.imshow(f"Segmentation {i}", segmented_mask)
#     cv.waitKey(0)
#     cv.destroyAllWindows()

# def segment_with_sam(image_path, centroid):
#     """Segment LEGO using SAM with centroid as input"""
#     # Load SAM model (choose "vit_h", "vit_l", or "vit_b")
#     sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")  # Ensure you have the right checkpoint
#     sam.to(device="cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available

#     # Create predictor
#     predictor = SamPredictor(sam)

#     # Load image
#     image = cv.imread(image_path)
#     image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
#     predictor.set_image(image_rgb)

#     # Provide the centroid as a segmentation prompt
#     input_point = np.array([centroid])  # Convert centroid to NumPy array
#     input_label = np.array([1])  # Label 1 for object segmentation

#     # Get segmentation mask
#     masks, scores, logits = predictor.predict(
#         point_coords=input_point,
#         point_labels=input_label,
#         multimask_output=True
#     )

#     return masks

###################################################
#Real time detection
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