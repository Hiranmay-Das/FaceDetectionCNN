import cv2
import os
import shutil
from mtcnn.mtcnn import MTCNN


# This line is for hiding the warning messages provided by Tensorflow.
# Level | Level for Humans | Level Description
#  -------|------------------|------------------------------------
#   0     | DEBUG            | [Default] Print all messages
#   1     | INFO             | Filter out INFO messages
#   2     | WARNING          | Filter out INFO & WARNING messages
#   3     | ERROR            | Filter out all messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def Open_CV(img, scaleFactor=1.3, minNeighbours=3, minSize=(0, 0), greyScale=True):
    """
    Reads an Image and detects the faces in this image using
    OpenCV's Haarcascade Classifier.

    --Parameters--
    img - Matrix of the type CV_8U containing an image where objects are detected.
    scaleFactor - Parameter specifying how much the image size is reduced at each image scale.
    minNeighbors - Parameter specifying how many neighbors each candidate rectangle should have to retain it.
    minSize - Minimum possible object size. Objects smaller than that are ignored.
    maxSize	- Maximum possible object size. Objects larger than that are ignored.
    If maxSize == minSize model is evaluated on single scale.
    greyScale - Flag to specify if image should be converted to grayscale before detection.

    """
    # try:
    #     img = cv2.imread(path)
    # except Exception:
    #     raise FileNotFoundError(
    #         "The path specified is incorrect or the file doesnot exist.")
    if greyScale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    CASCADE_PATH = [
        "A:\\Desktop\\Misc\\Python\\Object Classification NN\\venv\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml",
        "A:\\Desktop\\Misc\\Python\\Object Classification NN\\venv\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_alt.xml",
        "A:\\Desktop\\Misc\\Python\\Object Classification NN\\venv\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_alt2.xml",
    ]
    faceCascade = cv2.CascadeClassifier(CASCADE_PATH[0])
    faces = faceCascade.detectMultiScale(
        img,
        scaleFactor=scaleFactor,
        minNeighbors=minNeighbours,
        minSize=minSize
    )
    return faces


def MT_CNN(img, min_face_size=20, scale_factor=0.709):
    """
    Reads an Image and detects the faces in this image using
    a Pre-Trained Multi-task Cascaded Convolutional Neural Networks for Face Detection,
    based on TensorFlow.

    --Parameters--
    img - Matrix of the type CV_8U containing an image where objects are detected.
    min_face_size: Minimum possible object size. Objects smaller than that are ignored.
    scale_factor: Parameter specifying how much the image size is reduced at each image scale.

    """
    model = MTCNN(min_face_size=min_face_size, scale_factor=scale_factor)
    faces = model.detect_faces(img)
    return list(map(lambda x: x['box'], faces))


def draw_bounding_boxes(img, faces, color=(0, 255, 0), thickness=2):
    """
    Draws Bounding Boxes around detected objects on the provided picture.

    --Parameters--
    img - Matrix of the type CV_8U containing an image where objects are detected.
    faces - A NumPy array containing the matrices of detected objects.
    color - Contains BRG color values for the border.
    thickness - Thickness of the Border.

    """
    for x, y, w, h in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), color, thickness)
    return img


def resize_img(img, required_size=(160, 160)):
    return cv2.resize(img, required_size)


def show_img(img, title='Title', time=0):
    """
    Plots the provided image using matplotlib.

    --Parameters--
    img - Matrix of the type CV_8U containing an image where objects are detected.
    title - Name of the Window of the Plot.
    time - Time the window waits before closing down. By default, it's 0 which means
           the window waits for a key press to close.

    """
    cv2.imshow(title, img)
    key = cv2.waitKey(time)
    cv2.destroyAllWindows()
    # ----EXPERIMENTAL----
    # NEEDS TO BE DELETED
    if key == 27:
        raise StopIteration("Esc key pressed during Image Slideshow.")


def show_faces(img, faces):
    """
    Plots out all the faces detected in an Image.

    --Parameters--
    img - Matrix of the type CV_8U containing an image where objects are detected.
    faces - A NumPy array containing the matrices of detected objects.

    """
    for i, face in enumerate(faces, start=1):
        x1, y1, w, h = face
        x2, y2 = x1 + w, y1 + h
        show_img(img[y1:y2, x1:x2], f"Face {i}", 1000)


def save_faces(img, faces, file_count, path):
    """

    Saves all the detected faces in the given path along with the original image.

    --Parameters--
    img - Matrix of the type CV_8U containing an image where objects are detected.
    faces - A NumPy array containing the matrices of detected objects.
    file_count - Index of the image from which the faces have been extracted.
    path - Directory path provided by the user where the faces will be saved.

    """

    image_with_boxes = draw_bounding_boxes(img.copy(), faces)
    cv2.imwrite(os.path.join(path, "Original.png"), image_with_boxes)

    for i, face in enumerate(faces, start=1):
        x1, y1, w, h = face
        x2, y2 = x1 + w, y1 + h
        face = img[y1:y2, x1:x2]
        face = resize_img(face)

        cv2.imwrite(os.path.join(
            path, f"{file_count} - {i}.png"), face)


def change_file_names(path):
    """
    Serializes all the files in the provided directory
    --Parameters--
    path - Path to the directory containing the files to be serialized.

    """
    img_count = 1
    for file in os.scandir(path):
        if os.path.isdir(file.path):
            continue
        os.rename(file.path, f"{path}\\Image - {img_count}.png")
        img_count += 1


def extract_faces():
    """
    Extracts faces from all the images present in the 'training' folder. 
    Cropped faces are stored in serialized folders inside the main image folder.
    The path is traced using the Working Directory.

    """
    working_directory = os.getcwd()
    img_folder = os.path.join(working_directory, 'dataset\\train')
    faces_folder = os.path.join(img_folder, 'faces')
    if os.path.exists(faces_folder):
        shutil.rmtree(faces_folder)
    os.mkdir(faces_folder)
    file_count = 1
    for file in os.scandir(img_folder):
        # If the current path is a directory, then skip.
        if file.is_dir():
            continue

        # Reading the Image to a NumPy array
        img = cv2.imread(file.path)

        ###################
        # DETECTING FACES #
        ###################

        # Detecting Faces using OpenCV's HaarCascade Classifier
        # faces = Open_CV(img)

        # Detecting Faces using MTCNN from http://github.com/ipazc/mtcnn
        faces = MT_CNN(img, min_face_size=60, scale_factor=0.9)
        faces_folder_path = os.path.join(faces_folder, f"Image-{file_count}")
        os.mkdir(faces_folder_path)
        save_faces(img, faces, file_count, path=faces_folder_path)
        file_count += 1

    os.system("clear" if os.name == "posix" else "cls")
    print("FACES EXTRACTED AND SAVED.")


if __name__ == "__main__":
    extract_faces()
