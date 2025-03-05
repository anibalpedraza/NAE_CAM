from keras.applications.xception import Xception, preprocess_input as preprocess_xception
from keras.applications.efficientnet import EfficientNetB0, preprocess_input as preprocess_efficientnet
from keras.applications.inception_v3 import InceptionV3, preprocess_input as preprocess_inceptionv3
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
import pathlib
import tensorflow as tf
from time import strftime
from os.path import join as fullfile
from os import makedirs

networkModelName = 'EfficientNetB0' # 'InceptionV3' 'EfficientNetB0' #'Xception'

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
basePath='C:\\Users\\Aniba\\Documents\\Code\\VISILAB\\Dataset_NAE_CAM_Cyano'
datasetPath=fullfile(basePath,'dataset_cyano_processed')
nClasses=5


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
else:
   print('Model not supported')
   exit()


# Build the dataset
trainDatagen = ImageDataGenerator(preprocessing_function=preprocess_function,
                                  validation_split=validationSplit)
train_generator = trainDatagen.flow_from_directory(datasetPath,
                                                    target_size=imageSize,
                                                    batch_size=batchSize,
                                                    seed=13,
                                                    class_mode='categorical',
                                                    subset='training')
validation_generator = trainDatagen.flow_from_directory(datasetPath,
                                                        target_size=imageSize,
                                                        batch_size=batchSize,
                                                        seed=13,
                                                        class_mode='categorical',
                                                        subset='validation')


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
logdir = fullfile(basePath,"logs", timestamp)
makedirs(logdir,exist_ok=True)

checkpoint_path = fullfile(basePath,"checkpoints", timestamp, "cp-{epoch:04d}.ckpt")
makedirs(fullfile(basePath,"checkpoints", timestamp),exist_ok=True)
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
outputModelName=fullfile(outputModelPath,timestamp+"_"+networkModelName+"_model_base.h5")
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
model = tf.keras.models.load_model(outputModelName)
#model.summary()
model.evaluate(validation_generator)

print('Training ended')

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

print('Done')
