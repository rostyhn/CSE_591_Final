import cv2
import numpy as np
import pandas as pd
import os
import argparse
import pickle
import tensorflow as tf
from tensorflow import keras
from skimage.measure import label
from skimage.measure import regionprops
import PIL
from PIL import Image
from tensorflow.keras import layers
from natsort import natsorted

def create_model(img_size, num_classes):
    def get_model(img_size, num_classes):
        inputs = keras.Input(shape=img_size + (3,))

        ### [First half of the network: downsampling inputs] ###

        # Entry block
        x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs) # 32-filters; kernel_size=3
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        previous_block_activation = x  # Set aside residual

        # Blocks 1, 2, 3 are identical apart from the feature depth.
        for filters in [64, 128, 256]:
            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

            # Project residual
            residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
                previous_block_activation
            )
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        ### [Second half of the network: upsampling inputs] ###

        for filters in [256, 128, 64, 32]:
            x = layers.Activation("relu")(x)
            x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.UpSampling2D(2)(x)

            # Project residual
            residual = layers.UpSampling2D(2)(previous_block_activation)
            residual = layers.Conv2D(filters, 1, padding="same")(residual)
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        # Add a per-pixel classification layer
        outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)  # MODIFY HERE

        # Define the model
        model = keras.Model(inputs, outputs)
        return model


    # Free up RAM in case the model definition cells were run multiple times
    keras.backend.clear_session()

    # Build model
    model = get_model(img_size, num_classes)   # MODIFY/CHECK HERE
    model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")
    return model

def load_images(image_width,image_height, preProcess, frameFolder):
    print("Extracting images; {height}x{width}".format(height=image_height,width=image_width))
    images = []

    lst = natsorted((
        [
            os.path.join(frameFolder, fname)
            for fname in os.listdir(frameFolder)
            if fname.endswith(".jpg")
        ]
    ))

    for idx, filename in enumerate(lst):        
            fp = os.path.join(frameFolder,filename)
            if os.path.isfile(fp):
                img = Image.open(fp).resize((image_height,image_width))
                arr = np.array(img)
                arr = preProcess(arr)
                arr = arr[None,:,:,:]
                images.append(arr)
                print("Image {count} out of {total}".format(count=idx+1,total=len(os.listdir(frameFolder))), end="\r")
    return images

def frameCapture():
    # Path to video file
    vidObj = cv2.VideoCapture(videopath)
    # GT indexing starts at 1 (?!?!?!)
    count = 1
    # checks whether frames were extracted
    success = 1
    if vidObj.isOpened():
        while success:
            success, image = vidObj.read()
            if success:
                name = "{filedir}{filename}_frame_{count}_GT.jpg".format(
                    filedir=filedir, filename=filename, count=count)
                print("Creating {name}".format(name=name), end="\r")
                cv2.imwrite(name, image)
            count += 1
    cv2.destroyAllWindows()
    
def extractFrames(rootDirectory, frameFolderName):
    # for each directory that contains a video and ground truth, makes a "frames"
    # folder containing all the extracted frames
    if os.path.isdir(curr_dir):
        frameFolder = os.path.join(curr_dir, frameFolderName)
        os.mkdir(frameFolder)

        frameFolder += os.path.sep
        for file in os.listdir(curr_dir):
            filepath = os.path.join(curr_dir, file)
            if os.path.isfile(filepath):
                FrameCapture(os.path.splitext(file)[0], filepath, frameFolder)

if __name__ == "__main__":
    """
     2. load dataset    
     3. if frames are not extracted, extract frames
     8. ask user if they want to train the model, or only classify (optional)
     4. run classification, save in array
     5. run segmentation
     6. loop through frames, save classification + segmentation onto video
     7. save video
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('dir', metavar='dir', type=str, help='folder where the dataset resides')
    parser.add_argument('--frameFolder', default='frames', help='folder where the extracted frames reside', type=str)
    parser.add_argument('--classificationModel', default='cnn', choices=['bag_of_words','cnn'])
    
    args = parser.parse_args()

    rootDirectory = os.path.join(os.getcwd(), args.dir)        
    rootDirContents = os.listdir(rootDirectory)
    
    if args.frameFolder not in rootDirContents:
        print("Folder {folder} Extracting frames".format(folder=args.frameFolder))
        extractFrames(rootDirectory, args.frameFolder)

    """
    bag of features results
                  precision    recall  f1-score   support

           0       0.97      0.97      0.97       322
           1       0.90      0.90      0.90        98

    accuracy                           0.95       420
    macro avg      0.93      0.93      0.93       420
    weighted avg   0.95      0.95      0.95       420
    """

    model = None    
    scriptLocation = os.path.dirname(os.path.realpath(__file__))    
    modelsLocation = os.path.join(scriptLocation, 'models')
    opts = {}

    modelLocation = os.path.join(modelsLocation, 'saved')
    model = keras.models.load_model(modelLocation)

    frameFolder = os.path.join(rootDirectory, args.frameFolder)

    # Project 1
    def proj1_preprocess(arr):
        arr = tf.keras.applications.xception.preprocess_input(arr)
        return arr
        

    images = load_images(480,712, proj1_preprocess, frameFolder)    
    predictions = []
    batch = []
    for idx, image in enumerate(images):
        pred = model(image, training=False)
        predictions.append(pred.numpy())
        print("Image {count} out of {total}".format(count=idx+1,total=len(images)), end="\r")    
    
    print("Image informativeness classification complete")

    # Create a basic model instance
    segModel = create_model((320,320), 2)

    # Load weights from .h5 file
    polypWeights = os.path.join(modelsLocation, 'polyps_segmentation_2.h5')
    segModel.load_weights(polypWeights)

    def polypPreprocess(m):
        return m
    
    images = load_images(320,320, polypPreprocess, frameFolder)

    def predicted_mask(pred):
        mask = np.argmax(pred, axis=-1)
        mask = np.expand_dims(mask, axis=-1)
        return mask

    predictionsFolder = os.path.join(rootDirectory, 'predictions')
    vp = os.path.join(predictionsFolder, 'video.mp4')
    out = cv2.VideoWriter(vp, cv2.VideoWriter_fourcc(*'mp4v'), 10, (320,320))    
    
    if not os.path.exists(predictionsFolder):
        os.makedirs(predictionsFolder)
    for idx,image in enumerate(images):
        pred = segModel(image)
        mask = predicted_mask(pred)
        mask = np.reshape(mask,(320,320))
        # finds all pixels with prop > 0.5
        binary = mask > 0.5
        la = label(binary)
        props = regionprops(la)
        image = np.reshape(image, (320,320,3))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        informativeness = predictions[idx]
        for prop in props:
            image = cv2.rectangle(image, (prop.bbox[1], prop.bbox[0]), (prop.bbox[3], prop.bbox[2]), (255, 0, 0), 2)

        predString = "Clear: {class_0}%, Blurry: {class_1}%".format(
            class_0=int(informativeness[0][0]*100),
            class_1=int(informativeness[0][1]*100))
            
        image = cv2.putText(image,
                            predString,
                            (20,40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255,255,255), 2, cv2.LINE_AA)

        out.write(image)
        # writes an image to the path instead
        # fn = "frame_{idx}.jpg".format(idx=idx)
        # fp = os.path.join(predictionsFolder, fn)
        # status = cv2.imwrite(fp, image)
        print("Image {count} out of {total}".format(count=idx+1,total=len(images), end="\r"))            
        
    out.release()
    
    
    
