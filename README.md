# CognateCNN
This project is a convolutional neural network (CNN) that analyzes self-generated images in a variety of languages to find etymological similarities. Specifically, the goal is to prove that computer vision can be used to identify cognates known to exist, and perhaps lead linguists to evidence of unknown cognates. For a complete project description, please read the included paper "Cognate Analysis Using CNN". This describes in detail the purpose, implementation, and success of the project.

# Installation Notes
This implementation requires the installation of a Python version >= 3.5. Specifically, Python 3.8 was used in writing this software. A minimum of 3.5 is required in order to make full use of the built-in pathlib module.
https://www.python.org/downloads/release

Python pip dependencies in this project include:
matplotlib
numpy
tensorflow (should be TensorFlow 2)
wikipedia
pillow
unidecode

This implementation can be run within any interface that can read Python files. For example, any IDE that supports Python such as PyCharm would work.
This implementation can also be run at the command line. One needs only to preface the ImageNeuralNet.py file with the Python 3 python.exe, either by referencing the fully qualified path to your Python 3 installation directory, or by adding the Python 3 python.exe to your Windows PATH. If using Linux, there are multiple ways you could reference it too, such as creating an alias to the python.exe like 'alias python="path/to/python.exe"

Note: This implementation has not been tested on a Linux machine, although there are no Windows dependencies that I am aware of.



The user should see the following files:
- ImageNeuralNet.py
- TestImageNeuralNet.py
- TestSingleImage.py

Each of these files will also create a local directory when run. These local directories will be used to store the generated image data.

ImageNeuralNet.py is the crux of this project. It accepts no parameters. When run it will generate approximately 45,000 - 55,000 images using the wikipedia pages specified in the code. It will then pass the image data into a convolutional neural network to be processed and save the weights of the CNN model to a checkpoints folder within the local directory.

TestImageNeuralNet.py is mainly used for testing the weights generated by ImageNeuralNet.py. It will load the CNN model from the checkpoints folder and run against wikipedia pages specified in the code.

TestSingleImage.py when run will prompt the user to input any word in any language. The code will then load the CNN model from the checkpoints folder and output what it believes is the language the word most likely belongs to among the possibilities of English, Spanish, German, and Italian.

The pre-trained CNN weights within the checkpoints folder have been omitted in order to save space, as they are 30-40GB large. These can be offered too though if requested.
