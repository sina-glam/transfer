from ultralytics import YOLO
import cv2

def predict_with_bbox_xy(model, source, show=False, conf=0.5):
    """
    Modified predict method that prints the x,y values of the bounding box.
    """
    # Use DetectionPredictor to get predictions
    detector = DetectionPredictor(model)
    frames = detector.predict(source)

    # Loop over frames and draw bounding boxes
    for frame in frames:
        for pred in frame.pred:
            # Get class label, confidence, and bounding box coordinates
            cls, score, bbox = pred.cls, pred.score, pred.box.xyxy
            
            # Print x and y values of bounding box
            x1, y1, x2, y2 = bbox
            print(f"x1={x1}, y1={y1}, x2={x2}, y2={y2}")

            # Draw bounding box on frame
            if score > conf:
                frame = frame.draw_box(bbox, label=model.names[int(cls)], color='red')

        # Show or save the image
        if show:
            cv2.imshow('YOLO', frame.img)
            cv2.waitKey(1)
    return frames

# Load the YOLOv5 model
model = YOLO("/Users/.../Desktop/bottleTest/best.pt")

# Use the modified predict method to get predictions and print x,y values of bounding box
predict_with_bbox_xy(model, source="0", show=True, conf=0.5)
