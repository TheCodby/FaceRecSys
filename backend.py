from flask import Flask, request, jsonify
import recognition_service
import cv2
import numpy as np
import sqlite3

app = Flask(__name__)


@app.route('/detect_face', methods=['POST'])
def detect_face():
    file = request.files['image']
    npimg = np.fromfile(file, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    rgb_img = img[:, :, ::-1]

    face_locations = recognition_service.face_locations(rgb_img)
    return jsonify(face_locations=face_locations)


@app.route('/add_face', methods=['POST'])
def add_face():
    file = request.files['image']
    npimg = np.fromfile(file, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    rgb_img = img[:, :, ::-1]

    face_locations = recognition_service.face_locations(rgb_img)
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("INSERT INTO face VALUES (?,?,?,?)",
              (face_locations[0][0], face_locations[0][1], face_locations[0][2], face_locations[0][3]))
    conn.commit()
    conn.close()
    return jsonify(face_locations=face_locations)


if __name__ == '__main__':
    app.run(port=5000)
