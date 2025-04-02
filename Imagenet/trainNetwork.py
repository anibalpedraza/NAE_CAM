from keras.api.applications.xception import Xception, preprocess_input as preprocess_xception
from keras.api.applications.efficientnet import EfficientNetB0, preprocess_input as preprocess_efficientnet
from keras.api.applications.inception_v3 import InceptionV3, preprocess_input as preprocess_inceptionv3
from keras.api.applications.convnext import ConvNeXtTiny, preprocess_input as preprocess_convnext
from keras.api.models import Model
from keras.api.layers import Dense, GlobalAveragePooling2D
#from keras.api.preprocessing.image import ImageDataGenerator
from keras.api.utils import image_dataset_from_directory
from keras.api.saving import load_model

import pathlib
import tensorflow as tf
from time import strftime
from os.path import join as fullfile
from os import makedirs

from Imagenet.testNetwork import main as testNetwork

def main(networkModelName, datasetPath, basePath, nClasses):
   # Show the experiment information in a single message
   print('Training the model with the following parameters:')
   print('Network model:', networkModelName)
   print('Dataset path:', datasetPath)
   print('Base path:', basePath)


   batchSize=16
   validationSplit=0.2
   sample_size = len(list(pathlib.Path(datasetPath).rglob('./*')))


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
   train_generator,validation_generator = image_dataset_from_directory(datasetPath,
                                                      image_size=imageSize,
                                                      batch_size=batchSize,
                                                      validation_split=validationSplit,
                                                      seed=13,
                                                      label_mode='categorical',
                                                      subset='both',
                                                      shuffle=True)
                                         
   # Apply the same preprocessing as the training dataset using map and autotune
   train_generator = train_generator.map(lambda x, y: (preprocess_function(x), y))
   validation_generator = validation_generator.map(lambda x, y: (preprocess_function(x), y))
   # Optimize performance
   AUTOTUNE = tf.data.AUTOTUNE
   train_generator = train_generator.prefetch(buffer_size=AUTOTUNE)
   validation_generator = validation_generator.prefetch(buffer_size=AUTOTUNE)

   # add a global spatial average pooling layer
   x = base_model.output
   x = GlobalAveragePooling2D()(x)
   # let's add a fully-connected layer
   x = Dense(1024, activation='relu')(x)
   # and a logistic layer
   predictions = Dense(nClasses, activation='softmax')(x)

   # this is the model we will train
   model = Model(inputs=base_model.input, outputs=predictions)

   # first: train only the top layers (which were randomly initialized)
   # i.e. freeze all convolutional InceptionV3 layers
   for layer in base_model.layers:
      layer.trainable = False

   # compile the model (should be done *after* setting layers to non-trainable)
   model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

   # train the model on the new data for a few epochs with a tensorboard callback and
   # saving the checkpoint model each epoch
   timestamp=strftime("%Y%m%d_%H%M%S")
   logdir = fullfile(basePath,"logs", timestamp+"_"+networkModelName)
   makedirs(logdir,exist_ok=True)

   checkpoint_path = fullfile(basePath,"checkpoints", timestamp+"_"+networkModelName, "cp-{epoch:04d}.keras")
   makedirs(fullfile(basePath,"checkpoints", timestamp+"_"+networkModelName),exist_ok=True)
   checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=False, save_best_only=False, monitor='val_loss', mode='min')

   tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)
   model.fit(train_generator,
            steps_per_epoch=(round(sample_size*(1-validationSplit)))//batchSize,
            epochs=5,
            validation_data=validation_generator,
            validation_steps=(round(sample_size*validationSplit))//batchSize,
            callbacks=[tensorboard_callback, checkpoint_callback],
            verbose=1)

   # Save the model
   outputModelPath=fullfile(basePath,"checkpoints")
   print('Saving the model after training')
   outputModelName=fullfile(outputModelPath,timestamp+"_"+networkModelName+"_model_base.keras")
   model.save(outputModelName)

   # at this point, the top layers are well trained and we can start fine-tuning
   # convolutional layers from inception V3. We will freeze the bottom N layers
   # and train the remaining top layers.

   # let's visualize layer names and layer indices to see how many layers
   # we should freeze:
   #for i, layer in enumerate(base_model.layers):
   #   print(i, layer.name)

   # we chose to train the top 2 inception blocks, i.e. we will freeze
   # the first 249 layers and unfreeze the rest:ç

   # Load the model and testing
   #model = tf.keras.models.load_model(outputModelName)
   model = load_model(outputModelName)
   #model.summary()
   model.evaluate(validation_generator)

   print('Training ended')
   '''
   if networkModelName == 'Xception':
      for layer in model.layers[:249]:
         layer.trainable = False
      for layer in model.layers[249:]:
         layer.trainable = True

   # we need to recompile the model for these modifications to take effect
   # we use SGD with a low learning rate
   model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.0001, momentum=0.9), 
               loss='categorical_crossentropy', metrics=['accuracy'])

   # we train our model again (this time fine-tuning the top 2 inception blocks
   # alongside the top Dense layers
   model.fit(train_generator,
            steps_per_epoch=(round(sample_size*(1-validationSplit)))//batchSize,
            epochs=2,
            validation_data=validation_generator,
            validation_steps=(round(sample_size*validationSplit))//batchSize,
            callbacks=[tensorboard_callback])

   # Save the model
   print('Saving the model after fine-tuning')
   outputModelNameFinetuned=fullfile(outputModelPath,timestamp+"_"+networkModelName+"_model_tinetuned.h5")
   model.save(outputModelNameFinetuned)

   # Load the model and testing
   model = tf.keras.models.load_model(outputModelNameFinetuned)
   model.summary()
   model.evaluate(validation_generator)
   '''
   print('Done')
   return timestamp

if __name__ == '__main__':
   networkModelName = ['ConvNeXtTiny','EfficientNetB0','InceptionV3','Xception'] #'EfficientNetB0' # 'InceptionV3' 'EfficientNetB0' #'Xception'

   # Load the dataset from custom directory and classes
   #basePath='D:\\Dataset_NAE_CAM'
   '''
   basePath='C:\\Users\\Aniba\\Documents\\Code\\VISILAB\\Dataset_NAE_CAM'
   datasetPath=fullfile(basePath,'DatasetMerge_PNG')
   nClasses=46
   basePath='C:\\Users\\Aniba\\Documents\\Code\\VISILAB\\dataset_lucia_di_yolo'
   datasetPath=fullfile(basePath,'dataset_processed')
   nClasses=10
   basePath='C:\\Users\\Aniba\\Documents\\Code\\VISILAB\\Dataset_NAE_CAM_Cyano'
   datasetPath=fullfile(basePath,'dataset_cyano_processed')
   nClasses=5
   basePath='C:\\Users\\Aniba\\Documents\\Code\\VISILAB\\Dataset_NAE_CAM_Biopsy\\Biopsy_4classes'
   datasetPath=fullfile(basePath,'dataset_biopsy_4classes_train_processed')
   nClasses=4'
   '''

   envPath='/datasets/' # Docker
   #envPath='C:\\Users\\Aniba\\Documents\\Code\\VISILAB' # Alienware
   #envPath='D:\\VISILAB\\NAE_CAM' # Kratos
   #envPath='D:\\NAE_CAM' # PC

   cyanoDatasetBasePath=fullfile(envPath,'Dataset_NAE_CAM_Cyano')
   cyanoDatasetPath=fullfile(cyanoDatasetBasePath,'dataset_cyano_processed')
   cyanoNClasses=5

   biopsyDatasetBasePath=fullfile(envPath,'Dataset_NAE_CAM_Biopsy','Biopsy_4classes')
   biopsyDatasetPath=fullfile(biopsyDatasetBasePath,'dataset_biopsy_4classes_train_processed')
   biopsyNClasses=4

   for networkModelName in networkModelName:
      for  datasetPath, basePath, nClasses in [(cyanoDatasetPath, cyanoDatasetBasePath, cyanoNClasses),
                                               (biopsyDatasetPath, biopsyDatasetBasePath, biopsyNClasses)]:
         timestamp=main(networkModelName, datasetPath, basePath, nClasses)
         testNetwork(timestamp,networkModelName, datasetPath, basePath)
