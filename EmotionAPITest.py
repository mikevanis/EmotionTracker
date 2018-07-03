from CameraController import CameraController
import cv2
import requests

KEY = 'b44e51ccaba84ecdb4925ea905684885'
ENDPOINT = 'https://westcentralus.api.cognitive.microsoft.com/face/v1.0/detect'

headers = {'Content-Type': 'application/octet-stream', 'Ocp-Apim-Subscription-Key': KEY}
params = {
    'returnFaceId': 'true',
    'returnFaceLandmarks': 'false',
    'returnFaceAttributes': 'age,gender,headPose,smile,facialHair,glasses,' +
    'emotion,hair,makeup,occlusion,accessories,blur,exposure,noise'
}

def recognize(image):
    r, buf = cv2.imencode(".jpg", image)
    image_data = bytearray(buf)
    try:
        response = requests.post(data=image_data, url=ENDPOINT, headers=headers, params=params)
        response
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
