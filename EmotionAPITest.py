from CameraController import CameraController
import cv2
import requests

KEY = '0c3c2f2a95d94470950462f484cc6cc7'
#ENDPOINT = 'https://westcentralus.api.cognitive.microsoft.com/face/v1.0/detect'
ENDPOINT = 'https://uksouth.api.cognitive.microsoft.com/face/v1.0/detect'

headers = {'Content-Type': 'application/octet-stream', 'Ocp-Apim-Subscription-Key': KEY}
params = {
    'returnFaceId': 'true',
    'returnFaceLandmarks': 'false',
    'returnFaceAttributes': 'age,gender,headPose,smile,facialHair,glasses,' +
    'emotion,hair,makeup,occlusion,accessories,blur,exposure,noise'
}

eye_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_eye.xml")
face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")

def draw_objects(objects, frame):
    for(x, y, w, h) in objects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return frame

def recognize(image):
    r, buf = cv2.imencode(".jpg", image)
    image_data = bytearray(buf)
    try:
        response = requests.post(data=image_data, url=ENDPOINT, headers=headers, params=params)
        analysis = response.json()
        print(analysis)
    except requests.exceptions.HTTPError:
        print("HTTP Error. Request failed.")


if __name__ == '__main__':
    # Create new CameraController instance
    cam = CameraController()
    cam.start()

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

                cv2.imshow("Output", current_frame)


            key = cv2.waitKey(10)

            if key == ord('a'):
                recognize(current_frame)

            elif key == ord('q'):
                cam.stop()
                cam.join()
                cv2.destroyAllWindows()
                break

    except KeyboardInterrupt:
        cam.stop()
        cam.join()
        cv2.destroyAllWindows()
        pass
