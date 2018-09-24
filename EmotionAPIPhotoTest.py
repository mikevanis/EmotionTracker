import cv2
import requests

KEY = '0c3c2f2a95d94470950462f484cc6cc7'
ENDPOINT = 'https://uksouth.api.cognitive.microsoft.com/face/v1.0/detect'

headers = {'Content-Type': 'application/octet-stream', 'Ocp-Apim-Subscription-Key': KEY}
params = {
    'returnFaceId': 'true',
    'returnFaceLandmarks': 'false',
    'returnFaceAttributes': 'age,gender,headPose,smile,facialHair,glasses,' +
    'emotion,hair,makeup,occlusion,accessories,blur,exposure,noise'
}

def recognize(image):
    r, buf = cv2.imencode(".jpg", image)
    image_data = bytes(buf)
    try:
        response = requests.post(data=image_data, url=ENDPOINT, headers=headers, params=params)
        analysis = response.json()
        print(analysis)
    except requests.exceptions.HTTPError:
        print("HTTP Error. Request failed.")


if __name__ == '__main__':
    img = cv2.imread("tests/detectiontest.jpg")
    recognize(img)
