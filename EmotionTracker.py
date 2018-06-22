from CameraController import CameraController
import cv2

eye_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_eye.xml")
face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")


def draw_objects(objects, frame):
    for(x, y, w, h) in objects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return frame


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

    except KeyboardInterrupt:
        cam.stop()
        cam.join()
        cv2.destroyAllWindows()
        pass
