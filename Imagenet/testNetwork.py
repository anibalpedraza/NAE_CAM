from keras.applications.xception import Xception, preprocess_input as preprocess_xception
from keras.applications.efficientnet import EfficientNetB0, preprocess_input as preprocess_efficientnet
from keras.applications.inception_v3 import InceptionV3, preprocess_input as preprocess_inceptionv3
from keras.preprocessing.image import ImageDataGenerator
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
    else:
        print('Model not supported')
        exit()

    # Build the dataset
    testDatagen = ImageDataGenerator(preprocessing_function=preprocess_function)
    test_generator = testDatagen.flow_from_directory(datasetPath,
                                                            target_size=imageSize,
                                                            batch_size=batchSize,
                                                            seed=13,
                                                            class_mode='categorical',
                                                            shuffle=False)

    classNames=test_generator.class_indices.keys()
    # Load the model
    print('Loading model...')
    logdir = fullfile(basePath,"logs", timestamp+"_"+networkModelName)
    outputModelPath=fullfile(basePath,"checkpoints")
    outputModelName=fullfile(outputModelPath,timestamp+"_"+networkModelName+"_model_base.h5")

    # Load the model and testing
    print('Testing model...')
    model = tf.keras.models.load_model(outputModelName)
    #model.summary()
    model.evaluate(test_generator)

    # Get confusion matrix
    print('Getting confusion matrix...')

    # Get the true classes
    y_true = test_generator.classes
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
    '''
    basePath='C:\\Users\\Aniba\\Documents\\Code\\VISILAB\\Dataset_NAE_CAM_Cyano'
    datasetPath=fullfile(basePath,'dataset_cyano_processed')
    nClasses=5
    '''
    nClasses=2
    fold='test' #'train'
    basePath='C:\\Users\\Aniba\\Documents\\Code\\VISILAB\\Dataset_NAE_CAM_Biopsy\\Biopsy_'+str(nClasses)+'classes'
    datasetPath=fullfile(basePath,'dataset_biopsy_'+str(nClasses)+'classes_'+fold+'_processed')
    

    networkModelName = 'EfficientNetB0' # 'InceptionV3' 'EfficientNetB0' #'Xception'
    timestamp=strftime("20250311_124455") # 4 classes: "20250311_125705"

    main(timestamp, networkModelName, datasetPath, basePath, nClasses)