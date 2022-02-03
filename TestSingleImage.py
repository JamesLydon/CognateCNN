# James Lydon
# CS 530 - 900
# Computer Vision

# Necessary imports
import os
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf  # Tensorflow version should necessarily be a GPU-enabled TensorFlow 2
import tensorflow.keras as keras
from PIL import Image, ImageDraw
from tensorflow.keras.layers import Dense, Flatten, Conv2D


def create_single_image():
    # Make sure image path exists
    singleimagedir = "singleimage/"
    for dirs in os.listdir(singleimagedir):
        shutil.rmtree(os.path.join(singleimagedir, dirs))
    Path("singleimage/").mkdir(exist_ok=True)
    Path("singleimage/en").mkdir(parents=True, exist_ok=True)
    Path("singleimage/es").mkdir(parents=True, exist_ok=True)
    Path("singleimage/de").mkdir(parents=True, exist_ok=True)
    Path("singleimage/it").mkdir(parents=True, exist_ok=True)
    wordstring = input("Enter a word: ")
    tag = input("Enter what language this word is (en, de, es, it): ")
    img = Image.new('RGB', (84, 84), color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((10, 10), str(wordstring), fill=(0, 0, 0))
    imgfilename = "singleimage/" + tag + "/" + str(wordstring) + ".png"
    img.save(imgfilename)


# Train our Neural Network using TensorFlow / Keras on the above images
def neural_learning():
    # Set variables for the Neural Network model
    data_dir = "singleimage"
    CLASS_NAMES = np.array(["en", "es", "de", "it"])
    image_count = sum(len(files) for _, _, files in os.walk(r'singleimage'))
    BATCH_SIZE = 32
    IMG_HEIGHT = 84
    IMG_WIDTH = 84
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    # Displays some of the images for you to see how they look labeled
    def show_batch(image_batch, label_batch):
        plt.figure(figsize=(10, 10))
        for n in range(1):
            plt.imshow(image_batch[n])
            plt.title(CLASS_NAMES[label_batch[n] == 1][0].title())
            plt.axis('off')
        plt.show()

    list_ds = tf.data.Dataset.list_files(str(data_dir + '/*/*'))
    # Prints 5 of the random files so you can see if things look right
    for f in list_ds.take(5):
        print(f.numpy())

    def get_label(file_path):
        # Convert the path to a list of path components
        parts = tf.strings.split(file_path, os.path.sep)
        # The second to last is the class-directory
        return parts[-2] == CLASS_NAMES

    def decode_img(img):
        # Convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_jpeg(img, channels=3)
        # Use `convert_image_dtype` to convert to floats in the [0,1] range.
        img = tf.image.convert_image_dtype(img, tf.float32)
        # Resize the image to the desired size.
        return tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])

    def process_path(file_path):
        label = get_label(file_path)
        # Load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = decode_img(img)
        return img, label

    labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    # Print 1 example of Image Shape and Label
    for image, label in labeled_ds.take(1):
        print("Image shape: ", image.numpy().shape)
        print("Label: ", label.numpy())

    # Modify the dataset so that it's ready to be fitted through the neural network
    def prepare_for_training(ds, cache=False):
        if cache:
            if isinstance(cache, str):
                ds = ds.cache(cache)
            else:
                ds = ds.cache()
        ds = ds.repeat()
        ds = ds.batch(BATCH_SIZE)
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds

    test_ds = prepare_for_training(labeled_ds)

    # Display examples of the images / labels in the training dataset
    image_batch, label_batch = next(iter(test_ds))
    show_batch(image_batch.numpy(), label_batch.numpy())

    # Sequential, relu are used as recommended by the research paper read.
    # Softmax is used to give us a "percentage" of how accurate the model is for a given word and language
    model = keras.Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                     activation='relu',
                     input_shape=(84, 84, 3)))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(4, activation='softmax'))

    STEPS_PER_EPOCH = np.ceil(image_count / BATCH_SIZE)

    # Categorical_crossentropy used to determine which language label is most appropriate
    # Adam optimizer used as recommended by Research Paper read
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=0.001),
                  metrics=['accuracy'])

    checkpoint_path = "checkpoints/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    # Loads the weights
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    model.load_weights(latest)

    # Re-evaluate the model
    loss, acc = model.evaluate(test_ds, verbose=2, steps=STEPS_PER_EPOCH)
    print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
    prediction = model.predict(test_ds, steps=STEPS_PER_EPOCH)
    print(prediction)


create_single_image()
neural_learning()
