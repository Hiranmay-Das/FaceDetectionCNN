import os
import cv2
from FaceCropper import MT_CNN, resize_img

from tkinter import Tk
from tkinter.filedialog import askopenfilename
from tkinter.messagebox import showerror
Tk().withdraw()


def get_photo():
    """
    Accepts a photo from the user and returns it as a numpy array
    """
    filename = askopenfilename(
        title="Select Test Image",
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;")],
        initialdir=os.getcwd()
    )
    if filename:
        image = cv2.imread(filename)
    else:
        showerror("Error", "Unable to open file")
        return None
    return image


def label_face(image, face, predicted, rect_color=(0, 255, 0), text_color=(0, 0, 0)):
    """
    Labels the face in the image with the predicted label
    image - the image to be labeled
    face - the face to be labeled
    predicted - the label to be applied
    rect_color - the color of the rectangle
    text_color - the color of the text
    """
    x, y, w, h = face
    font_size, font_thiccness = 0.6, 2
    l, b = cv2.getTextSize(
        predicted, cv2.FONT_HERSHEY_SIMPLEX, font_size, font_thiccness)[0]
    cv2.rectangle(image, (x, y), (x+w, y+h), rect_color, 2)
    cv2.rectangle(image, (x-1, y), (x+max(l, w)+1, y-b-10), rect_color, -1)
    cv2.putText(image, predicted, (x, y-6),
                cv2.FONT_HERSHEY_SIMPLEX, font_size, text_color, font_thiccness)
    return image


def identify_faces(image, faces_loc, faces_data):
    """
    Identifies the faces in the image and labels them
    faces_data - the data of the faces
    faces_loc - the location of the faces
    """
    # model = load_model()
    for face_loc, face_data in zip(faces_loc, faces_data):
        # predicted = model.predict(face_data)
        predicted = "Unknown"
        image = label_face(
            image,
            face_loc,
            predicted,
            rect_color=(46, 0, 230),
            text_color=(255, 255, 255)
        )
    return image


def locate_faces(image):
    """
    Detects faces in the image and returns their locations
    image - the image to be analyzed
    """
    faces_loc = MT_CNN(image, min_face_size=60, scale_factor=0.747)
    return faces_loc


def encode_faces(image, faces_loc):
    """
    Crops the faces from the image and returns their data
    image - the image to be analyzed
    faces_loc - the location of the faces
    """
    faces_data = []
    for (x, y, w, h) in faces_loc:
        face = image[y:y+h, x:x+w]
        face = resize_img(face)
        faces_data.append(face)

    return faces_data


def main():
    image = get_photo()
    faces_loc = locate_faces(image)
    faces_data = encode_faces(image, faces_loc)
    image = identify_faces(image, faces_loc, faces_data)
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


main()
