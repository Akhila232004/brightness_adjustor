import cv2
import numpy as np
import screen_brightness_control as sbc

# Capture from webcam
cap = cv2.VideoCapture(0)

prev_center_x = None
brightness = 50  # initial brightness
sbc.set_brightness(brightness)

print("🖐️ Move your hand left/right to adjust brightness. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)

    _, thresh = cv2.threshold(blurred, 70, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 3000:
            x, y, w, h = cv2.boundingRect(largest)
            center_x = x + w // 2
            center_y = y + h // 2
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)

            if prev_center_x is not None:
                delta_x = center_x - prev_center_x
                if abs(delta_x) > 20:
                    if delta_x > 0:
                        brightness = min(100, brightness + 5)
                    else:
                        brightness = max(0, brightness - 5)
                    sbc.set_brightness(brightness)

            prev_center_x = center_x

    # Show brightness on screen
    cv2.putText(frame, f'Brightness: {brightness}%', (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

    cv2.imshow('Gesture-Based Brightness Control', frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
