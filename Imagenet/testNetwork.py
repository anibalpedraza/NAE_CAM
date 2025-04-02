from keras.api.applications.xception import Xception, preprocess_input as preprocess_xception
from keras.api.applications.efficientnet import EfficientNetB0, preprocess_input as preprocess_efficientnet
from keras.api.applications.inception_v3 import InceptionV3, preprocess_input as preprocess_inceptionv3
from keras.api.applications.convnext import ConvNeXtTiny, preprocess_input as preprocess_convnext
#from keras.api.preprocessing.image import ImageDataGenerator
from keras.api.utils import image_dataset_from_directory
from keras.api.saving import load_model

import tensorflow as tf

from time import strftime
from os.path import join as fullfile
from os import makedirs

from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import pandas as pd

def main(timestamp, networkModelName, datasetPath, basePath):
    
    # Load the dataset from custom directory and classes
    batchSize=16
    
    # create the base pre-trained model
    if networkModelName == 'Xception':
        base_model = Xception(weights='imagenet', include_top=False)
        preprocess_function = preprocess_xception
        imageSize=(299, 299)
    elif networkModelName == 'EfficientNetB0':
        base_model = EfficientNetB0(weights="imagenet", include_top=False)
        preprocess_function = preprocess_efficientnet
        imageSize=(224, 224)
    elif networkModelName == 'InceptionV3':
        base_model = InceptionV3(include_top=False, weights="imagenet")
        preprocess_function = preprocess_inceptionv3
        imageSize=(299, 299)
    elif networkModelName == 'ConvNeXtTiny':
        base_model = ConvNeXtTiny(weights='imagenet', include_top=False)
        preprocess_function = preprocess_convnext
        imageSize=(224, 224)
    else:
        print('Model not supported')
        exit()

    # Build the dataset
    _,test_generator = image_dataset_from_directory(datasetPath,
                                                            image_size=imageSize,
                                                            batch_size=batchSize,
                                                            validation_split=0.2,
                                                            seed=13,
                                                            label_mode='categorical',
                                                            subset='both',
                                                            shuffle=True)
    classNames=test_generator.class_names
    # Apply the same preprocessing as the training dataset using map and autotune
    test_generator = test_generator.map(lambda x, y: (preprocess_function(x), y),)
    AUTOTUNE = tf.data.AUTOTUNE
    test_generator = test_generator.prefetch(buffer_size=AUTOTUNE)

    # Load the model
    print('Loading model...')
    logdir = fullfile(basePath,"logs", timestamp+"_"+networkModelName)
    outputModelPath=fullfile(basePath,"checkpoints")
    outputModelName=fullfile(outputModelPath,timestamp+"_"+networkModelName+"_model_base.keras")

    # Load the model and testing
    print('Testing model...')
    model = load_model(outputModelName)
    #model.summary()
    #model.evaluate(test_generator)

    # Get confusion matrix
    print('Getting confusion matrix...')

    # Get the true classes
    #y_true = test_generator.classes
    y_true = np.concatenate([y for x, y in test_generator], axis=0)
    y_true = np.argmax(y_true, axis=1)
    # Get the predicted classes
    y_pred = model.predict(test_generator)
    y_pred = np.argmax(y_pred, axis=1)

    # Get the confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classNames)))
    #print(cm)

    # Get the classification report
    #print(classification_report(y_true, y_pred, target_names=test_generator.class_indices))

    # Get the accuracy
    print('Accuracy: ', accuracy_score(y_true, y_pred))

    # Save confusion matrix and report to csv files
    resultsPath=fullfile(basePath,"results",timestamp+"_"+networkModelName)
    makedirs(resultsPath,exist_ok=True)
    print('Saving confusion matrix and report...')
    cm_df = pd.DataFrame(cm, index = classNames, columns = classNames)
    cm_df.to_csv(fullfile(resultsPath,'confusion_matrix.csv'))
    report = classification_report(y_true, y_pred, labels=range(len(classNames)), target_names=classNames, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(fullfile(resultsPath,'classification_report.csv'))


    print('Done')

if __name__ == "__main__":
    # Load the dataset from custom directory and classes
    #basePath='D:\\Dataset_NAE_CAM'
    '''
    basePath='C:\\Users\\Aniba\\Documents\\Code\\VISILAB\\Dataset_NAE_CAM'
    datasetPath=fullfile(basePath,'DatasetMerge_PNG')
    nClasses=46
    '''
    '''
    basePath='C:\\Users\\Aniba\\Documents\\Code\\VISILAB\\dataset_lucia_di_yolo'
    datasetPath=fullfile(basePath,'dataset_processed')
    nClasses=10
    '''
    
    envPath='/datasets/' # Docker
    #envPath='C:\\Users\\Aniba\\Documents\\Code\\VISILAB' # Alienware
    #envPath='D:\\VISILAB\\NAE_CAM' # Kratos
    #envPath='D:\\NAE_CAM' # PC

    basePath=fullfile(envPath,'Dataset_NAE_CAM_Cyano')
    datasetPath=fullfile(basePath,'dataset_cyano_processed')
    nClasses=5
    
    '''
    nClasses=4
    fold='test' #'train'
    basePath='C:\\Users\\Aniba\\Documents\\Code\\VISILAB\\Dataset_NAE_CAM_Biopsy\\Biopsy_'+str(nClasses)+'classes'
    datasetPath=fullfile(basePath,'dataset_biopsy_'+str(nClasses)+'classes_'+fold+'_processed')
    timestamp=strftime("20250311_125705") # 4 classes: "20250311_125705" # 2 classes: "20250311_124455"
    '''
    networkModelName = 'ConvNeXtTiny' #'InceptionV3' 'EfficientNetB0' #'Xception'
    timestamp='20250326_165847'
    
    
    main(timestamp, networkModelName, datasetPath, basePath)