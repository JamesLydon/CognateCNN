# James Lydon
# CS 530 - 900
# Computer Vision

# Necessary imports
import os
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf  # Tensorflow version should necessarily be a GPU-enabled TensorFlow 2
import tensorflow.keras as keras
import wikipedia
from PIL import Image, ImageDraw
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from unidecode import unidecode

# This data structure contains the list of Wikipedia pages the application will run through to collect words from
# each language. "en, es, de, it" are the abbreviations Wikipedia uses. The pages chosen to be used in the data
# structure are pages which possess a diverse and lengthy vocabulary in that particular language.
languages = {
    'en': ['Computer', 'New York City', 'Animal', 'Julius Caesar', 'Nancy_Pelosi', 'George_Washington', 'Philosophy',
           'Religion'],
    'es': ['Computadora', 'Madrid', 'Animalia', 'Julio César', 'Pedro_Sánchez', 'Isabel_I_de_Castilla', 'Filosofía',
           'Religión'],
    'de': ['Computer', 'Berlin', 'Tier', 'Gaius Iulius Caesar', 'Angela_Merkel', 'Otto_von_Bismarck', 'Philosophie',
           'Religion'],
    'it': ['Computer', 'Roma', 'Animalia', 'Gaio Giulio Cesare', 'Giuseppe Conte', 'Leonardo da Vinci', 'Filosofia',
           'Religione']
}


def create_neural_data():
    # Make sure images paths exist
    Path("images/en").mkdir(parents=True, exist_ok=True)
    Path("images/es").mkdir(parents=True, exist_ok=True)
    Path("images/de").mkdir(parents=True, exist_ok=True)
    Path("images/it").mkdir(parents=True, exist_ok=True)

    # Keep the words we use relatively short or medium length only
    maxletters = 12
    # Loop through the languages one by one and generate dictionary of words for each
    for language in languages.keys():
        print('Generating dictionary for ' + language)
        languagedict = make_dict(language, maxletters)
        # Convert the dictionary of words into images
        convert_dict_to_images(languagedict, maxletters, language)


# Create dictionary of vocabulary words per each language
def make_dict(language, maxlength):
    wikipedia.set_lang(language)
    lst = []
    # For each Wikipedia page, grab only the real content on the page and convert the text to ASCII
    for topic in languages[language]:
        page = wikipedia.WikipediaPage(topic)
        content = page.content
        content = unidecode(content)
        wordsintopic = clean_words(content, maxlength)
        for word in wordsintopic:
            lst.append(word)
    return lst


# Clean bad words. Remove symbols, set everything to lowercase, cut words that are too long
def clean_words(content, maxlength):
    words = re.sub(r'[^a-zA-Z ]', '', content)
    lower = words.lower()
    wordlist = lower.split()
    shortwords = []
    for word in wordlist:
        if len(word) <= maxlength:
            shortwords.append(word)
    return shortwords


# Use Pillow to convert the ASCII word into an appropriate image in the images/ folder
def convert_dict_to_images(dic, maxletters, tag):
    for word in dic:
        if len(word) <= maxletters:
            wordstring = str(word)
            img = Image.new('RGB', (84, 84), color=(255, 255, 255))
            d = ImageDraw.Draw(img)
            d.text((10, 10), wordstring, fill=(0, 0, 0))
            imgfilename = "images/" + str(tag) + "/" + wordstring + ".png"
            # I'm not sure why but Python os package breaks when writing this one specific word
            if "con.png" in imgfilename:
                continue
            img.save(imgfilename)


# Train our Neural Network using TensorFlow / Keras on the above images
def neural_learning():
    # Set variables for the Neural Network model
    data_dir = "images"
    CLASS_NAMES = np.array(["en", "es", "de", "it"])
    image_count = sum(len(files) for _, _, files in os.walk(r'images'))
    BATCH_SIZE = 32
    IMG_HEIGHT = 84
    IMG_WIDTH = 84
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    # Displays some of the images for you to see how they look labeled
    def show_batch(image_batch, label_batch):
        plt.figure(figsize=(10, 10))
        for n in range(25):
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
    def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
        if cache:
            if isinstance(cache, str):
                ds = ds.cache(cache)
            else:
                ds = ds.cache()
        ds = ds.shuffle(buffer_size=shuffle_buffer_size)
        ds = ds.repeat()
        ds = ds.batch(BATCH_SIZE)
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds

    train_ds = prepare_for_training(labeled_ds)
    test_ds = train_ds.take(8000)
    train_ds = train_ds.skip(8000)

    # Display examples of the images / labels in the training dataset
    image_batch, label_batch = next(iter(train_ds))
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

    STEPS_PER_EPOCH = np.ceil(image_count - 8000 / BATCH_SIZE)
    VALIDATION_STEPS = np.ceil(8000 / BATCH_SIZE)

    # Categorical_crossentropy used to determine which language label is most appropriate
    # Adam optimizer used as recommended by Research Paper read
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=0.001),
                  metrics=['accuracy'])

    # Save model per epoch to checkpoints/ directory
    checkpoint_path = "checkpoints/cp-{epoch:04d}.ckpt"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_weights_only=True,
        period=1)

    model.save_weights(checkpoint_path.format(epoch=0))
    model.fit(train_ds,
              batch_size=32,
              epochs=10,
              verbose=1,
              validation_data=test_ds,
              validation_steps=VALIDATION_STEPS,
              steps_per_epoch=STEPS_PER_EPOCH,
              callbacks=[cp_callback])

    # Evaluate and print the results
    score = model.evaluate(test_ds, verbose=1)  # changed from 0 to 1, should be okay?
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    print("success?")


create_neural_data()
neural_learning()
