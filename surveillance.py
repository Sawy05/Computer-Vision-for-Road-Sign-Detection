import cv2
import numpy as np

# =========================
# VIDEO SOURCE
# =========================
cap = cv2.VideoCapture("video.mp4")  # or 0 for webcam

# Read first two frames
ret, frame1 = cap.read()
ret, frame2 = cap.read()

if not ret:
    print("Error: Video not found or cannot be read.")
    exit()

# Resize for performance
frame1 = cv2.resize(frame1, (640, 480))
frame2 = cv2.resize(frame2, (640, 480))

# =========================
# VARIABLES
# =========================
counter = 0
line_y = 300
object_id = 0
counted_ids = set()

# =========================
# MAIN LOOP
# =========================
while cap.isOpened():

    # Motion detection (frame difference)
    diff = cv2.absdiff(frame1, frame2)

    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thresh = cv2.threshold(
        blur,
        25,
        255,
        cv2.THRESH_BINARY
    )

    # Noise removal
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Draw virtual line
    cv2.line(frame1, (0, line_y), (640, line_y), (0, 0, 255), 2)

    # Process each contour
    for contour in contours:

        area = cv2.contourArea(contour)

        if area < 1000:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        cx = int(x + w / 2)
        cy = int(y + h / 2)

        # Simple classification
        if area < 3000:
            label = "Person"
        else:
            label = "Car"

        # Draw bounding box
        cv2.rectangle(
            frame1,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame1,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        # Counting logic (simple line crossing)
        if cy > line_y:
            if object_id not in counted_ids:
                counter += 1
                counted_ids.add(object_id)

        object_id += 1

        # ROI for edge detection
        roi = frame1[y:y+h, x:x+w]

        if roi.size != 0:

            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            edges = cv2.Canny(gray_roi, 50, 150)

            # Hough Lines
            lines = cv2.HoughLinesP(
                edges,
                1,
                np.pi / 180,
                50,
                minLineLength=20,
                maxLineGap=5
            )

            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(roi, (x1, y1), (x2, y2), (255, 0, 0), 1)

    # Show counter
    cv2.putText(
        frame1,
        f"Count: {counter}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    # Display output
    cv2.imshow("Smart Surveillance System", frame1)
    cv2.imshow("Threshold", thresh)

    # Update frames
    frame1 = frame2

    ret, frame2 = cap.read()

    if not ret:
        break

    frame2 = cv2.resize(frame2, (640, 480))

    # Exit condition
    key = cv2.waitKey(1)
    if key == 27 or key == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()