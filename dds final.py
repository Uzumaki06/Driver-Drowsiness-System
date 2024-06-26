import cv2
import dlib
from scipy.spatial import distance
import winsound
from imutils import face_utils

# Function to calculate the eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Function to calculate the mouth aspect ratio (MAR)
def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[2], mouth[10])
    B = distance.euclidean(mouth[4], mouth[8])
    C = distance.euclidean(mouth[0], mouth[6])
    mar = (A + B) / (2.0 * C)
    return mar

# Constants for eye aspect ratio (EAR) thresholds
EAR_THRESHOLD = 0.25
EAR_CONSEC_FRAMES = 48

# Constants for mouth aspect ratio (MAR) thresholds
MAR_THRESHOLD = 0.7
MAR_CONSEC_FRAMES = 10
YAWN_COUNTER_THRESHOLD = 6

# Initialize dlib's face detector and create a facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize the frame counters
frames_counter = 0
blink_counter = 0
yawn_counter = 0
drowsy = False
yawn_ended = True

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray)

    for face in faces:
        # Get the facial landmarks
        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        # Draw a rectangle around the face
        (x, y, w, h) = face_utils.rect_to_bb(face)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw the landmarks on the face
        for (x, y) in landmarks:
            cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

        # Extract the left and right eye coordinates
        left_eye = landmarks[42:48]
        right_eye = landmarks[36:42]

        # Extract the mouth coordinates
        mouth = landmarks[48:68]

        # Calculate the eye aspect ratios (EARs)
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)

        # Average the EARs for both eyes
        avg_ear = (left_ear + right_ear) / 2.0

        # Calculate the mouth aspect ratio (MAR)
        mar = mouth_aspect_ratio(mouth)

        # Check if the average EAR is below the threshold
        if avg_ear < EAR_THRESHOLD:
            frames_counter += 1
            if frames_counter >= EAR_CONSEC_FRAMES:
                # If eyes are closed for a sufficient number of frames, trigger drowsiness
                drowsy = True
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # Play a beep sound
                winsound.Beep(1000, 500)  # Adjust the frequency and duration as needed
        else:
            frames_counter = 0
            drowsy = False

        # Check if the mouth aspect ratio (MAR) exceeds the threshold
        if mar > MAR_THRESHOLD:
            if yawn_ended:
                yawn_counter += 1
                yawn_ended = False
            if yawn_counter >= YAWN_COUNTER_THRESHOLD:
                # If mouth is open for a sufficient number of frames, trigger yawn detection
                cv2.putText(frame, "YAWN DETECTED!", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # Play a beep sound
                winsound.Beep(1000, 500)  # Adjust the frequency and duration as needed
        else:
            yawn_ended = True

        # Draw the calculated eye aspect ratio (EAR) and mouth aspect ratio (MAR) on the frame
        cv2.putText(frame, "EAR: {:.2f}".format(avg_ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "MAR: {:.2f}".format(mar), (300, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display the yawn counter on the frame
        cv2.putText(frame, "Yawns: {}".format(yawn_counter), (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("Driver Drowsiness and Yawn Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
