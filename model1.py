from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, MaxPool2D, Flatten, Dense, Rescaling, Dropout, RandomFlip, RandomRotation, RandomZoom
from tensorflow.keras.applications import ResNet152V2
import random

SEED = random.randint(0, 1000)
IMG_WIDTH = 160
IMG_HEIGHT = 160
BATCH_SIZE = 32
NUMBER_OF_CLASSES = 6
EPOCHS = 10

def load_data():
    train_data_dir = "dataset//train/labelled data"
    train_data = image_dataset_from_directory(
        train_data_dir,
        seed=SEED,
        validation_split=0.2,
        subset="training",
        label_mode="categorical",
        image_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        color_mode='rgb',
        shuffle=True
    )

    validation_data = image_dataset_from_directory(
        train_data_dir,
        seed=SEED,
        validation_split=0.2,
        subset="validation",
        label_mode="categorical",
        image_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        color_mode='rgb',
        shuffle=True
    )
    
    # FOR VIEWING IMAGES
    # import cv2
    # class_names = train_data.class_names
    # print(class_names)
    # for images, labels in train_data.take(1):
    #     for i in range(9):
    #         cv2.imshow(class_names[labels[i]], images[i].numpy().astype("uint8"))
    #         cv2.waitKey(0)
    #         cv2.destroyAllWindows()

    return train_data, validation_data


def load_model():
    train_data, validation_data = load_data()
    model = Sequential()
    model.add(Rescaling(1./255, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))
    model.add(RandomFlip("horizontal", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))
    model.add(RandomRotation(0.2, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))
    model.add(RandomZoom(0.25, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))
    model.add(Convolution2D(64, 
                            kernel_size=(5, 5), 
                            strides=(1, 1), 
                            input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), 
                            activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Convolution2D(64, 
                            kernel_size=(5, 5), 
                            strides=(1, 1), 
                            activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(NUMBER_OF_CLASSES, activation='softmax'))
    model.compile(loss='categorical_crossentropy', 
                  optimizer='adam', 
                  metrics=["accuracy"])
    model.fit(
        train_data.shuffle(buffer_size=100, reshuffle_each_iteration=True).repeat(),
        steps_per_epoch=32,
        epochs=EPOCHS,
        validation_data=validation_data.repeat(),
        validation_steps=10)
    return model

if __name__ == "__main__":
    model = load_model()
    model.save("model.h5")