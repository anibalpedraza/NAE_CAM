from __future__ import print_function

import gradCamInterface
import auxiliarFunctions as aux

import os
import errno
import tensorflow as tf
from tensorflow import keras
#Mas ataques: https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/attacks/evasion.html#fast-gradient-method-fgm
from art.estimators.classification import TensorFlowV2Classifier

# ------------------------ Funciones auxiliares ---------------------------------
def train_step(model, images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

# DEPRECATED: executeGradCam(num, classifier, epsilon, n_iter):
# https://stackoverflow.com/questions/66182884/how-to-implement-grad-cam-on-a-trained-network

def executeGradCam(orig, adv) :
    # Prepare image
    list_img = []  # Orig, adversaria
    list_img.append(orig)
    list_img.append(adv)
    plot_img = []

    last_conv_layer_name = aux.getLastConvLayerName(NetworkModelName)
    # Generate class activation heatmap
    for ind in range(0, len(list_img)):
        img_array = gradCamInterface.get_img_array(list_img[ind].data)
        heatmap = gradCamInterface.make_gradcam_heatmap(img_array, model, last_conv_layer_name)
        list_img[ind].addHeatmap(heatmap)

        # Display heatmap. Ya esta entre 0-255
        gradCam_img = gradCamInterface.display_gradcam(list_img[ind].data, heatmap)

        plot_img.append(keras.preprocessing.image.array_to_img(list_img[ind].data))
        plot_img.append(gradCam_img)

    orig.addHeatmap(list_img[0].heatmap)
    adv.addHeatmap(list_img[1].heatmap)
    aux.printResultsPerImage(orig, adv)
    print("     ------------------")
    return plot_img, list_img

# ------------------------ Constantes ---------------------------------------
NUM_CLASSES = 1000 #imagenet=1000
#EFFICIENTNETB0 IMG_SIZE = (224, 224)#IMG_SHAPE = (224, 224, 3)
IMG_SIZE = (299, 299)
IMG_SHAPE = (299, 299, 3)
LR = 0.01 #Learning Rate usado en el optimizador
NUM_IMG = 50 #Cantidad de imagenes de test
TOTAL_IMG = 1000 #Cantidad de imagenes de las que se disponen, imagenet=50000
IMG_PATH = "C:/Users/User/TFG-repository/webcam_gradcam/ImageNetWebcam/waterBottle_xception/frames_raw/"
#EXECUTION_ID = "WebcamData_01" #Se usará para no sustituir variables de distintas ejecuciones
EXECUTION_ID = "WebcamData_Xception"
#IMG_PATH = "C:/Users/User/TFG-repository/Imagenet/movil/"#cambiar parametros de entrada de loadImages segun si son de imagenet o no
realID='n04557648'

#EPSILON = [20000, 30000]
ATTACK_NAME = ['FastGradientMethod']
NetworkModelName = 'Xception'

# ------------------------ Código principal ---------------------------------
# Load model: CNN -> EfficientNetB0
model = aux.getNetworkModel(NetworkModelName, IMG_SHAPE)
model.trainable = False
optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
classifier = TensorFlowV2Classifier(model=model, clip_values=(0, 1), nb_classes=NUM_CLASSES, input_shape=IMG_SHAPE, loss_object=loss_object, train_step=train_step)

#Load Images
randomVector = aux.generateRandomVector(NUM_IMG, TOTAL_IMG)
x_test, img_test = aux.loadImages(IMG_PATH, randomVector, size= IMG_SIZE, unclassified_images=True, realID=realID, networkName='Xception')# Quitar unclassified_images y realID para imagenet
#Si createImages = True: cargará las imagenes originales desde la carpeta y generará las adversarias de cero
#Si unclassified_images = True: cargará las imagenes que no son de imagenet y por tanto no estan dentro de una carpeta con el valor de su ID

#Generate Adversarials
img_adv=[]
for atck in range(0, len(ATTACK_NAME)) :
    individual_atck = []
    for num in range(0, len(img_test)):
        img_adv.append(aux.generateAnAdversarialImage(img_test[num], x_test[num], ATTACK_NAME[atck], classifier, isImagenet=False))

    individual_atck = img_adv[atck:atck+len(img_test)]
    filename = "Adv_Images_AttackMethod_" + ATTACK_NAME[atck] + ".pkl"
    aux.saveVariable(individual_atck, filename)
#Hasta aqui tenemos una lista de objetos imagenes para originales y adversarias, en ambas se ha predicho ya la clase

#GRAD CAM
# Remove last layer's softmax
model.layers[-1].activation = None #efficientnetb0
#print(model.summary())

for atck in range(0, len(ATTACK_NAME)):
    for num in range(0, NUM_IMG):
        img_figure, list_img_data = executeGradCam(img_test[num], img_adv[num])
        aux.saveResults(img_figure, list_img_data, EXECUTION_ID)
        aux.plotDifferenceBetweenImages(img_test[num], img_adv[num], EXECUTION_ID)
aux.calculatePercentageNaturalAdversarial(img_test)

# Save variables
try :
    os.mkdir('variables')
except OSError as e :
    if e.errno != errno.EEXIST :
        raise
aux.saveVariable(img_test, "variables/%s_testImages_efficientnetB0_random%simages.pkl" % (EXECUTION_ID, NUM_IMG))
aux.saveVariable(img_adv, "variables/%s_adversarials_images_atcks_%s" % (EXECUTION_ID, ATTACK_NAME) + ".pkl")
#https://stackoverflow.com/questions/66182884/how-to-implement-grad-cam-on-a-trained-network