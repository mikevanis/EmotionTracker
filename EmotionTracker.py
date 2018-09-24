#!/usr/bin/python
from CameraController import CameraController
import cv2
import requests
import numpy as np
import operator
import imutils
import RPi.GPIO as gpio
import os

os.chdir("/home/pi/EmotionTracker")

KEY = '0c3c2f2a95d94470950462f484cc6cc7'
ENDPOINT = 'https://uksouth.api.cognitive.microsoft.com/face/v1.0/detect'

headers = {'Content-Type': 'application/octet-stream', 'Ocp-Apim-Subscription-Key': KEY}
params = {
    'returnFaceId': 'true',
    'returnFaceLandmarks': 'false',
    'returnFaceAttributes': 'age,gender,headPose,smile,facialHair,glasses,' +
    'emotion,hair,makeup,occlusion,accessories,blur,exposure,noise'
}

test_data = [{u'faceId': u'e4abb911-ed15-4920-9160-eab699d2e6c8',
              u'faceRectangle': {u'width': 116, u'top': 71, u'height': 116, u'left': 121},
              u'faceAttributes': {u'emotion':
                                      {u'sadness': 0.0, u'neutral': 0.999, u'contempt': 0.0, u'disgust': 0.0,
                                       u'anger': 0.0, u'surprise': 0.0, u'fear': 0.0, u'happiness': 0.0},
                                  u'noise': {u'noiseLevel': u'low', u'value': 0.0},
                                  u'gender': u'male',
                                  u'age': 31.0, u'makeup': {u'lipMakeup': False, u'eyeMakeup': False},
                                  u'accessories': [{u'confidence': 1.0, u'type': u'glasses'}],
                                  u'facialHair': {u'sideburns': 0.6, u'moustache': 0.6, u'beard': 0.9},
                                  u'hair': {u'invisible': False, u'hairColor':
                                      [{u'color': u'black', u'confidence': 0.86},
                                       {u'color': u'brown', u'confidence': 0.82},
                                       {u'color': u'gray', u'confidence': 0.79},
                                       {u'color': u'other', u'confidence': 0.31},
                                       {u'color': u'blond', u'confidence': 0.24},
                                       {u'color': u'red', u'confidence': 0.06}],
                                        u'bald': 0.34},
                                  u'headPose': {u'yaw': -3.3, u'roll': -2.6, u'pitch': 0.0},
                                  u'blur': {u'blurLevel': u'low', u'value': 0.0},
                                  u'smile': 0.0, u'exposure': {u'exposureLevel': u'goodExposure', u'value': 0.62},
                                  u'occlusion': {u'mouthOccluded': False, u'eyeOccluded': False, u'foreheadOccluded': False},
                                  u'glasses': u'ReadingGlasses'}}]

eye_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_eye.xml")
face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
data_image = np.zeros((1080, int(1920/2), 3), np.uint8)
analyse = False

def draw_objects(objects, frame):
    for(x, y, w, h) in objects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return frame

def recognize(image):
    r, buf = cv2.imencode(".jpg", image)
    image_data = bytes(buf)
    try:
        response = requests.post(data=image_data, url=ENDPOINT, headers=headers, params=params)
        analysis = response.json()
        print(analysis)
        print(len(analysis))
        return analysis
    except requests.exceptions.HTTPError:
        print("HTTP Error. Request failed.")
        return None

def draw_data(data_object, x=10, y=30, w=int(1920/2), h=1080, padding_bottom=30):
    output_image = np.zeros((h, w, 3), np.uint8)
    # Get first two faces
    if data_object.__len__() > 1:
        data_object = data_object[:2]

    for index, face in enumerate(data_object):
        draw_text(output_image, "Face " + str(index), x, y, bold=True)
        y = y + padding_bottom
        y = y + padding_bottom

        face_attributes = face.get('faceAttributes', {})

        # Emotions
        emotions = face_attributes.get('emotion', {})
        emotions_sorted = list(reversed(sorted(emotions.items(), key=lambda kv: kv[1])))
        first_emotion, confidence = emotions_sorted[0]
        draw_text(output_image, "Emotion: " + first_emotion + " " + str(confidence*100) + "%", x, y, bold=True)
        del emotions_sorted[0]
        y = y + padding_bottom
        y = draw_attributes(output_image, x, y, padding_bottom, emotions_sorted)
        y = y + padding_bottom

        # Gender
        gender_val = face_attributes['gender']
        draw_text(output_image, "Gender: " + gender_val, x, y, bold=True)
        y = y + padding_bottom
        y = y + padding_bottom

        # Age
        age_val = face_attributes['age']
        draw_text(output_image, "Age: " + str(age_val), x, y, bold=True)
        y = y + padding_bottom
        y = y + padding_bottom

        # Hair
        hair_attributes = face_attributes['hair']
        hair_color = hair_attributes['hairColor']
        hair_dict = {}
        for c in hair_color:
            key = c['color']
            val = c['confidence']
            hair_dict[key] = val
        hair_sorted = list(reversed(sorted(hair_dict.items(), key=lambda kv: kv[1])))
        first_hair, confidence = hair_sorted[0]
        draw_text(output_image, "Hair: " + first_hair + " " + str(confidence*100) + "%", x, y, bold=True)
        del hair_sorted[0]
        y = y + padding_bottom
        y = draw_attributes(output_image, x, y, padding_bottom, hair_sorted)
        y = y + padding_bottom

        # Accessories
        accessories_attributes = face_attributes['accessories']
        accessories_dict = {}
        for a in accessories_attributes:
            key = a['type']
            val = a['confidence']
            accessories_dict[key] = val
        accessories_sorted = list(reversed(sorted(accessories_dict.items(), key=lambda kv: kv[1])))
        if accessories_sorted.__len__() > 0:
            first_accessory, confidence = accessories_sorted[0]
            draw_text(output_image, "Accessories: " + first_accessory + " " + str(confidence*100) + "%", x, y, bold=True)
            del accessories_sorted[0]
            y = y + padding_bottom
            y = draw_attributes(output_image, x, y, padding_bottom, accessories_sorted)

        # Facial hair
        facial_hair_attributes = face_attributes['facialHair']
        facial_hair_sorted = list(reversed(sorted(facial_hair_attributes.items(), key=lambda kv: kv[1])))
        first_facial_hair, confidence = facial_hair_sorted[0]
        draw_text(output_image, "Facial Hair: " + first_facial_hair + " " + str(confidence * 100) + "%", x, y, bold=True)
        del facial_hair_sorted[0]
        y = y + padding_bottom
        y = draw_attributes(output_image, x, y, padding_bottom, facial_hair_sorted)
        y = y + padding_bottom

        # Smile
        smile_val = face_attributes['smile']
        draw_text(output_image, "Smile: " + str(smile_val), x, y, bold=True)
        y = y + padding_bottom
        y = y + padding_bottom

        #cv2.imshow("data", output_image)
        return output_image

def draw_attributes(img, x, y, padding_bottom, list_of_attributes, break_after=4):
    attributes_strings = list()
    attributes_strings.append("")
    strings_index = 0
    words_index = 0
    for a in list_of_attributes:
        a_key, a_val = a
        attributes_strings[strings_index] = attributes_strings[strings_index] + a_key + " " + str(a_val * 100) + "%  "
        words_index = words_index + 1
        if words_index >= 4:
            strings_index = strings_index + 1
            attributes_strings.append("")
            words_index = 0

    for s in attributes_strings:
        draw_text(img, s, x, y)
        y = y + padding_bottom

    return y


def draw_text(image, text, x, y, bold=False):
    if bold is True:
        cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)

#draw_data(test_data)

if __name__ == '__main__':
    # Create new CameraController instance
    cam = CameraController()
    cam.start()

    cv2.namedWindow("Output", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Output", cv2.WND_PROP_FULLSCREEN, 1)

    gpio.setmode(gpio.BCM)
    gpio.setwarnings(False)

    gpio.setup(17, gpio.IN, gpio.PUD_UP)
    gpio.setup(23, gpio.OUT)

    light = gpio.PWM(23, 50)
    light.start(100)

    try:
        while True:
            current_frame = cam.get_image()

            if current_frame is not None:
                gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
                eye_objects = eye_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 5,
                                                           minSize = (30, 30), flags = cv2.CASCADE_SCALE_IMAGE)
                face_objects = face_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 5,
                                                           minSize = (30, 30), flags = cv2.CASCADE_SCALE_IMAGE)
                if len(eye_objects):
                    current_frame = draw_objects(eye_objects, current_frame)
                if len(face_objects):
                    current_frame = draw_objects(face_objects, current_frame)

                camera_side = np.zeros((1080, int(1920/2), 3), np.uint8)
                resized_frame = imutils.resize(current_frame, width=int(1920/2), height=1080)

                x_offset = 0
                y_offset = 100
                camera_side[y_offset:y_offset + resized_frame.shape[0], x_offset:x_offset + resized_frame.shape[1]] = resized_frame
                if analyse is True:
                    results = recognize(current_frame)
                    if results is not None:
                        if len(results) > 0:
                            data_image = draw_data(results)
                            light.ChangeDutyCycle(100)
                        else:
                            print("No faces found")
                    else:
                        print("Results is none")
                    analyse = False

                if data_image is not None:
                    if data_image.shape[0] == 1080 and data_image.shape[1] == int(1920/2):
                        output = np.hstack((data_image, camera_side))
                else:
                    print("Data image is none.")


                cv2.imshow("Output", output)

            key = cv2.waitKey(10)

            if key == ord('a'):
                analyse = True

            if gpio.input(17) == False:
                analyse = True
                light.ChangeDutyCycle(0)

            elif key == ord('q'):
                cam.stop()
                cam.join()
                cv2.destroyAllWindows()
                break

    except KeyboardInterrupt:
        cam.stop()
        cam.join()
        cv2.destroyAllWindows()
        #gpio.cleanup()
        pass
