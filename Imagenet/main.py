from __future__ import print_function

import gradCamInterface
import auxiliarFunctions as aux

import os
import errno
import tensorflow as tf
from tensorflow import keras
#Mas ataques: https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/attacks/evasion.html#fast-gradient-method-fgm
from art.estimators.classification import TensorFlowV2Classifier

from os.path import join as fullfile

# ------------------------ Funciones auxiliares ---------------------------------
def train_step(model, images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

# DEPRECATED: executeGradCam(num, classifier, epsilon, n_iter):
# https://stackoverflow.com/questions/66182884/how-to-implement-grad-cam-on-a-trained-network

def executeGradCam(model, orig, adv) :
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


# ------------------------ Código principal ---------------------------------
def main(BASE_PATH, fovFolder, fovName, NUM_CLASSES, realID, ATTACK_NAME, NetworkModelName, timestamp):
    # Initial variables

    #EFFICIENTNETB0 IMG_SIZE = (224, 224)#IMG_SHAPE = (224, 224, 3)
    IMG_SIZE = (224, 224)#(299, 299)
    IMG_SHAPE = (224, 224, 3)#(299, 299, 3)
    LR = 0.01 #Learning Rate usado en el optimizador

    #IMG_PATH = "C:/Users/User/TFG-repository/Imagenet/movil/"#cambiar parametros de entrada de loadImages segun si son de imagenet o no
    #IMG_PATH = fullfile(BASE_PATH, 'frames_raw_crop_2') # 'frames_raw_crop' # 'frames_raw'
    #EXECUTION_ID = "WebcamData_01" #Se usará para no sustituir variables de distintas ejecuciones
    IMG_PATH = fullfile(BASE_PATH, fovFolder, fovName, 'images')
    kindModel = 'base' #'finetuned'
    networkCheckpointName= timestamp+'_'+NetworkModelName+'_model_'+kindModel+'.h5'

    #EXECUTION_ID = timestamp+"WebcamData_"+NetworkModelName+"_02"#"_01"
    EXECUTION_ID = timestamp+"_OpenFlexure_"+NetworkModelName+"_"+fovName#"_01"

    # Get number of images in IMG_PATH
    nImages = len([name for name in os.listdir(IMG_PATH) if os.path.isfile(os.path.join(IMG_PATH, name))])
    NUM_IMG = nImages #90 #Cantidad de imagenes de test
    TOTAL_IMG = nImages #90 #Cantidad de imagenes de las que se disponen, imagenet=50000

    # Load model: CNN -> EfficientNetB0
    modelPath=fullfile(BASE_PATH,'checkpoints',networkCheckpointName)
    model = aux.getNetworkModel(NetworkModelName,customModel=True, modelPath=modelPath)
    model.trainable = False
    optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    classifier = TensorFlowV2Classifier(model=model, clip_values=(0, 1), nb_classes=NUM_CLASSES, input_shape=IMG_SHAPE, loss_object=loss_object, train_step=train_step)

    #Load Images
    randomVector = aux.generateRandomVector(NUM_IMG, TOTAL_IMG)
    x_test, img_test = aux.loadImages(IMG_PATH, randomVector, size= IMG_SIZE,
                                    unclassified_images=True, realID=realID, networkName=NetworkModelName)# Quitar unclassified_images y realID para imagenet
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
            img_figure, list_img_data = executeGradCam(model,img_test[num], img_adv[num])
            aux.saveResults(img_figure, list_img_data, EXECUTION_ID)
            aux.plotDifferenceBetweenImages(img_test[num], img_adv[num], EXECUTION_ID)
    aux.calculatePercentageNaturalAdversarial(img_test)

    # Save variables
    try :
        os.mkdir(fullfile('results','variables'))
    except OSError as e :
        if e.errno != errno.EEXIST :
            raise
    aux.saveVariable(img_test, fullfile('results','variables',"%s_testImages_%s_random%simages.pkl" % 
                                        (EXECUTION_ID, NetworkModelName, NUM_IMG)))
    aux.saveVariable(img_adv, fullfile('results','variables',"%s_adversarials_images_atcks_%s" % 
                                    (EXECUTION_ID, ATTACK_NAME) + ".pkl"))
    #https://stackoverflow.com/questions/66182884/how-to-implement-grad-cam-on-a-trained-network

if __name__ == '__main__':
    # ------------------------ Constantes ---------------------------------------
    #BASE_PATH='D:\\Dataset_NAE_CAM'
    #BASE_PATH='C:\\Users\\Aniba\\Documents\\Code\\VISILAB\\Dataset_NAE_CAM'
    #BASE_PATH='C:\\Users\\Aniba\\Documents\\Code\\VISILAB\\Dataset_NAE_CAM_Cyano'
    BASE_PATH='D:\\Dataset_NAE_CAM_Cyano'

    fovFolder='FOVs_Alberto_v3'
    fovName=['FOV_Dolichospermum1','FOV_Phormidium1','FOV_Phormidium2',
             'FOV_Phormidium3','FOV_Phormidium4','FOV_Phormidium5',
             'FOV_Raphidiopsis1','FOV_Tolypothrix1','FOV_Tolypothrix2'] #'FOV_Raphidiopsis1'

    NUM_CLASSES = 5 #(Cyano) # Diatoms: 46 #imagenet=1000
    realID=[0,2,2,
            2,2,2,
            3,4,4] #3 #(3: Raphidiopsis) #(4: Tolypothrix) # 13: 'Cymbella' #Default: 'n04557648'

    #EPSILON = [20000, 30000]
    ATTACK_NAME = ['FastGradientMethod']#,'ProjectedGradientDescent','BoundaryAttack']
    NetworkModelName = 'EfficientNetB0' # 'InceptionV3' # 'EfficientNetB0' #'Xception'
    timestamp = '20250225_171006' #'20240730_122157' #'20240730_130240' #'20240730_122157' #'20240729_134137'
    
    # Execute main function
    for indexExp in range(0, len(fovName)):
        main(BASE_PATH, fovFolder, fovName[indexExp], NUM_CLASSES, realID[indexExp],
             ATTACK_NAME, NetworkModelName, timestamp)