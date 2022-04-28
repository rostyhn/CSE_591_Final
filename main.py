import cv2
import numpy as np
import pandas as pd
import os
import argparse
import pickle
import tensorflow as tf
import skimage
from skimage.feature import daisy
from skimage import data
#from skimage.measure import label
#from skimage.measure import regionprops
#from natsort import natsorted

def get_daisy_descriptor(img):
    descriptor = daisy(img,visualize=False)    
    count = descriptor.shape[0] * descriptor.shape[1]
    descriptor = descriptor.reshape(count, descriptor.shape[2])
    return descriptor

def extract_image_features(img, model):
    features = get_daisy_descriptor(img)
        
    image_clustering = model.predict(features)
    
    frequency_counts = pd.DataFrame(image_clustering,columns=['cnt'])['cnt'].value_counts()
    vec = np.zeros(model.n_clusters)
    for k in frequency_counts.keys():
        vec[k] = frequency_counts[k]
    
    vec /= np.linalg.norm(vec)
    return list(vec)

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

def modelPredictionStrategy(model, modelName, images, opts):
    if modelName == 'bag_of_words':
        predictions = []
        clustering = opts['clustering']
        for idx, image in enumerate(images):            
            predictions.append(model.predict([extract_image_features(image, clustering)]))
            print(predictions)
            print("Image {count} out of {total}".format(count=idx+1,total=len(images)), end="\r")
    return predictions
    
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
    parser.add_argument('--classificationModel', default='bag_of_words', choices=['bag_of_words','cnn'])
    
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
    # TODO add CNN
    if args.classificationModel == 'bag_of_words':
        modelLocation = os.path.join(modelsLocation, 'bag_of_words.pickle')        
        with open (modelLocation, 'rb') as f:            
            model = pickle.load(f)
        clusteringLocation = os.path.join(modelsLocation, 'clustering.pickle')
        with open(clusteringLocation, 'rb') as f: 
            opts['clustering'] = pickle.load(f)

    frameFolder = os.path.join(rootDirectory, args.frameFolder)
    images = []

    print("Extracting images...")    
    for idx, filename in enumerate(os.listdir(frameFolder)):        
            fp = os.path.join(frameFolder,filename)
            if os.path.isfile(fp):
                img = skimage.io.imread(fp)                
                gr_img = skimage.color.rgb2gray(img)                
                gr_img /= gr_img.max()
                images.append(gr_img)
                print("Image {count} out of {total}".format(count=idx+1,total=len(os.listdir(frameFolder))), end="\r")
                
    predictions = modelPredictionStrategy(model, args.classificationModel, images, opts)
    print(predictions)
    
