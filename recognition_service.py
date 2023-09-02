import face_recognition
import cv2
import numpy as np
import csv
import sys
from datetime import datetime
import encode
import os

cache_file = "known_face_encodings.npy"
if not os.path.exists(cache_file):
    encode.load_known_faces()

known_face_encodings = np.load(
    "known_face_encodings.npy", allow_pickle=True)
known_face_names = np.load("known_face_names.npy", allow_pickle=True)
attendance_list = []


def encode_faces(rgb_small_frame, face_locations):
    if len(face_locations) != 0:
        return face_recognition.face_encodings(
            rgb_small_frame, face_locations)
    else:
        return []


def recognize_face(face_encoding):
    matches = face_recognition.compare_faces(
        known_face_encodings, face_encoding)
    name = "Unknown"
    if True in matches:
        first_match_index = matches.index(True)
        name = known_face_names[first_match_index]
    return name


def added_to_attendance(name):
    if name not in attendance_list:
        attendance_list.append(name)
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        with open(f"{now.strftime('%Y-%m-%d')}.csv", "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([name, current_time])


def draw_face_locations(frame, top, right, bottom, left, name):
    top *= 4
    right *= 4
    bottom *= 4
    left *= 4
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
    cv2.rectangle(frame, (left, bottom - 35),
                  (right, bottom), (0, 0, 255), cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(frame, name, (left + 6, bottom - 6),
                font, 0.5, (255, 255, 255), 1)


def main():
    reencode = input("Re-encode faces? (y/n): ")
    if reencode == "y":
        encode.load_known_faces()
    elif reencode == "n":
        pass
    else:
        print("Invalid input")
        sys.exit(1)
    face_encodings = []
    video_capture = cv2.VideoCapture(0)

    while True:
        _, frame = video_capture.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

        face_locations = face_recognition.face_locations(rgb_small_frame)
        # If face_locations and face_encodings are not the same length, then recognize unknown faces
        if len(face_locations) != len(face_encodings):
            face_encodings = encode_faces(rgb_small_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            name = recognize_face(face_encoding)
            draw_face_locations(frame, top, right, bottom, left, name)
            added_to_attendance(name)

        cv2.imshow('FaceRecSys', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
