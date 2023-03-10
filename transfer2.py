import cv2
import numpy as np
import torch

# Set the object class ID to "bottle"
object_class_id = 0

# Load the PyTorch YOLOv8 model
model = torch.load('yolov8.pt')

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Resize the frame to the model's input size
    resized_frame = cv2.resize(frame, (model.input_size, model.input_size))

    # Convert the frame to a PyTorch tensor
    tensor_frame = np.transpose(resized_frame, (2, 0, 1))
    tensor_frame = torch.from_numpy(tensor_frame).float().div(255.0).unsqueeze(0)

    # Pass the frame to the model and get the predictions
    with torch.no_grad():
        output = model(tensor_frame.cuda())

    # Parse the output predictions to get the bounding box coordinates
    boxes = []
    for box in output[0]:
        # Get the class probabilities
        class_probs = box[5:].tolist()
        # Get the class with maximum probability
        class_id = class_probs.index(max(class_probs))
        # If the class is not "bottle", skip the box
        if class_id != object_class_id:
            continue
        # Get the box coordinates
        x, y, w, h, confidence = box[:5].tolist()
        # Convert the box coordinates from YOLO format to pixel format
        x = int((x - w / 2) * tensor_frame.shape[3])
        y = int((y - h / 2) * tensor_frame.shape[2])
        w = int(w * tensor_frame.shape[3])
        h = int(h * tensor_frame.shape[2])
        boxes.append((x, y, w, h))

    # Draw the bounding box on the original frame
    for box in boxes:
        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('frame', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Release the webcam and close the windows
cap.release()
cv2.destroyAllWindows()
