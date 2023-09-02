import os
import numpy as np
import face_recognition


def load_known_faces():
    known_face_encodings = []
    known_face_names = []
    for filename in os.listdir("faces"):
        try:
            image = face_recognition.load_image_file(f"faces/{filename}")
            encoding = face_recognition.face_encodings(image)[0]
            if len(encoding) != 128:
                print(f"Failed to encode {filename}")
                continue
            known_face_encodings.append(encoding)
            known_face_names.append(filename.split(".")[0])
            np.save("known_face_encodings.npy", known_face_encodings)
            np.save("known_face_names.npy", known_face_names)
        except Exception as e:
            print(e)
            print(f"Failed to encode {filename}")
            continue
