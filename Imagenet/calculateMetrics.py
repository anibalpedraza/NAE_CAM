import numpy as np

from tensorflow import keras
from sklearn.metrics import mean_squared_error #con 3chnn no funciona
from skimage.metrics import structural_similarity as ssim
from statistics import median
import cv2
from scipy.stats import shapiro

import gradCamInterface
import auxiliarFunctions as aux
import auxiliarMetricsFunctions as mf

from os.path import join as fullfile
from os import listdir

#https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html
#https://www.jmp.com/es_es/statistics-knowledge-portal/t-test/two-sample-t-test.html
# ------------------------ Constantes ---------------------------------------
ATCK_NAME = ["FastGradientMethod"]#"BoundaryAttack" ["BoundaryAttack", "FastGradientMethod", "ProjectedGradientDescent"]HopSkipJump
NUM_ATCKS = len(ATCK_NAME) #Numero de ataques distintos que se usaron cuando se guardaron las imagenes

timestamp='20250225_171006'
networkName = "EfficientNetB0" #"Xception" #EfficientNetB0
fovName='FOV_Phormidium1'
DATA_ID=timestamp+"_OpenFlexure_"+networkName+'_'+fovName
# Search all pickles files matching baseName in the results/variables folder
pickleFileNamesList=[]
for file in listdir(fullfile('results','variables')):
    if file.startswith(DATA_ID) and file.endswith('.pkl'):
        pickleFileNamesList.append(fullfile('results','variables',file))
# Load all files and merge them into a single list
sorted_data = []
for pickleFileName in pickleFileNamesList:
    sorted_data += aux.loadVariable(pickleFileName)

#DATA_PATH = "C:/Users/User/TFG-repository/Imagenet/case1/ArtificialAdversarial-BoundaryAttack_%s/" % (DATA_ID)#"D:/TFG_VISILAB_FOTOS/case01_waterBottle/results_%s/variablesIndividuales_Test_%s/" % (DATA_ID, DATA_ID)
#sorted_data, name_list = aux.loadImagesSorted(DATA_PATH, NUM_ATCKS)
name_list = []
for i in range(0, len(sorted_data)):
    name_list.append(sorted_data[i].name)
NUM_IMG = len(sorted_data)

# ------------------------ Operaciones --------------------------------------
calculate_metrics = True #tarda 'poco'
execute_Histogram = False #tarda mucho
execute_BoxPlot = False

if calculate_metrics == True:
    metricsName = ["Nombre Imagen", "Ataque", "Epsilon", "Predicción", "Media", "Media normalizada", "Mediana",
                   "Varianza", "Desviación Típica", "Centroide Máximo", "Distancia centroides máximos",
                   "Centroide Mínimo", "Distancia centroides mínimos", "Dif de medias", "Diferencia Norma Mascara",
                   "Diferencia Norma Imagen", "MSE", "PSNR", "SSIM"]
    aux.createCsvFile(DATA_ID+"_metrics.csv", metricsName)

bins = 24
mean_heatmap_array_advNatural = []
mean_heatmap_array_advArtificial = []
mean_heatmap_array_original = []
freq_heatmap_array_advNatural = []
freq_heatmap_array_advArtificial = []
freq_heatmap_array_original = []
if len(ATCK_NAME) != 1 :
    valoresMetricas = mf.initializeVariablesToSave(ATCK_NAME, sorted_data[0]) #Dentro del vector ira Orig - AdvNatural - Adv1 - Adv2...

for num in range(0, NUM_IMG):
    gray_heatmap = gradCamInterface.display_gray_gradcam(sorted_data[num].data, sorted_data[num].heatmap,
                                                         superimposed=False)
    gray_heatmap = cv2.cvtColor(gray_heatmap, cv2.COLOR_RGB2GRAY)  # plt.imshow(gray, cmap='gray')
    heatmap_array = gray_heatmap.flatten()
    metricsValue = []
    metricsValue = mf.writeDataImageInCSV(metricsValue, sorted_data[num]) #Se escribe "Nombre Imagen", "Ataque", "Epsilon", "Prediccion"
    """result = shapiro(heatmap_array)
    if result[1] < 0.05:
        print('La muestra no sigue una distribucion normal')
    else:
        print('No se tienen suficientes evidencias para asegurar que es una distribución normal')"""
    if calculate_metrics == True:
        if metricsValue[1] != "Original" :
            closer_orig_img = mf.searchCloserOriginalImage(sorted_data, name_list, num)
            heatmap_ref = gradCamInterface.display_gray_gradcam(closer_orig_img.data, closer_orig_img.heatmap,
                                                                superimposed=False)
            heatmap_ref = cv2.cvtColor(heatmap_ref, cv2.COLOR_RGB2GRAY)
        else:
            heatmap_ref = []

        metricsValue.append(round(gray_heatmap.mean(), 2)) #"Media"
        metricsValue.append(round(gray_heatmap.mean()/255, 2)) #"Media normalizada"
        metricsValue.append(round(median(heatmap_array), 2)) #"Mediana"
        metricsValue.append(round(gray_heatmap.var(), 2)) #"Varianza"
        metricsValue.append(round(gray_heatmap.std(), 2)) #"Desviación Típica"
        mf.writeCentroidsInCSV(metricsValue, gray_heatmap, 215, heatmap_ref) #"Centroide Maximo","Distancia centroides maximos","Centroide Minimo","Distancia centroides minimos"

        if metricsValue[1] != "Original": #Si no es la imagen original
            # La norma es la distancia euclidea
            metricsValue.append(abs(round(gray_heatmap.mean()-heatmap_ref.mean(), 2))) #"Dif de medias"
            metricsValue.append(round(np.linalg.norm(gray_heatmap - heatmap_ref), 2)) #"Norma Mascara"
            metricsValue.append(round(np.linalg.norm(sorted_data[num].data - closer_orig_img.data), 2)) #"Norma Imagen"
            metricsValue.append(round(mean_squared_error(gray_heatmap, heatmap_ref), 2)) #Mean Squared Error (MSE)
            metricsValue.append(round(cv2.PSNR(gray_heatmap, heatmap_ref), 2)) #Peak Signal-to-Noise Ratio (PSNR)
            metricsValue.append(round(ssim(gray_heatmap, heatmap_ref, data_range=255, channel_axis=-1), 2))#SSIM. The higher the value, the more "similar" the two images are.
            #Desviacion media absoluta?
            #VIF (Visual Information Fidelity - ?Factor de inflación de la varianza)https://github.com/pavancm/Visual-Information-Fidelity---Python https://github.com/baidut/matLIVE
            #IFC (Information Fidelity Criterion)
            #MAD (Most Apparent Distortion) https://github.com/dingkeyan93/IQA-optimization/blob/master/IQA_pytorch/MAD.py
            #L2
        else:
            param=len(metricsName)-len(metricsValue)
            for index in range(0, param):
                metricsValue.append("-")



        aux.addRowToCsvFile(DATA_ID+"_metrics.csv", metricsName, metricsValue)
        if len(ATCK_NAME) != 1 :
            mf.saveMetricsInVariable(ATCK_NAME, metricsValue, valoresMetricas)

    #heatmap_array_sinMenorQue25 = [x for x in heatmap_array if x > 25]
    #https://joserzapata.github.io/courses/python-ciencia-datos/visualizacion/
    #se ejecuta por imagen
    if execute_Histogram == True:
        aux.saveHistogram(heatmap_array, sorted_data[num], DATA_ID)#Parece distribucion geometrica/exponencial xd
    if execute_BoxPlot == True:
        aux.saveBoxPlot(heatmap_array, sorted_data[num], DATA_ID)

    if len(ATCK_NAME) == 1 :  # No es un vector por lo que puedo generar las graficas del ataque correspondiente
        meanPerBin, freqPerBin = mf.meanFreqPerBin(bins, heatmap_array)
        if metricsValue[1] != "Original":
            if metricsValue[1] == "Adv. Natural":
                mean_heatmap_array_advNatural += meanPerBin
                freq_heatmap_array_advNatural.append(freqPerBin)

            else :
                mean_heatmap_array_advArtificial += meanPerBin
                freq_heatmap_array_advArtificial.append(freqPerBin)
        else:
            mean_heatmap_array_original += meanPerBin
            freq_heatmap_array_original.append(freqPerBin)
if len(ATCK_NAME) == 1 :  # No es un vector por lo que puedo generar las graficas del ataque correspondiente
    freq500_Orig, std_orig = mf.meanFreqTotalImgPerBin(freq_heatmap_array_original)
    freq500_AdvNat, std_nat = mf.meanFreqTotalImgPerBin(freq_heatmap_array_advNatural)
    freq500_AdvArt, std_art = mf.meanFreqTotalImgPerBin(freq_heatmap_array_advArtificial)
    mean500_Orig, x = mf.meanFreqPerBin(bins, mean_heatmap_array_original)
    mean500_AdvNat, x = mf.meanFreqPerBin(bins, mean_heatmap_array_advNatural)
    mean500_AdvArt, x = mf.meanFreqPerBin(bins, mean_heatmap_array_advArtificial)
    aux.saveBarWithError(mean500_Orig, freq500_Orig, std_orig, "originales", DATA_ID)
    aux.saveBarWithError(mean500_AdvNat, freq500_AdvNat, std_nat, "adv. naturales", DATA_ID)
    aux.saveBarWithError(mean500_AdvArt, freq500_AdvArt, std_art, "adv. artificiales,", DATA_ID,
                         ATCK_NAME[0])
    mf.createDataFrameToPlot(freq500_Orig, freq500_AdvNat, freq500_AdvArt,
                          std_orig, std_nat, std_art, DATA_ID, atck=ATCK_NAME[0])
    aux.saveMeanLineWithError(mean500_Orig, mean500_AdvNat, mean500_AdvArt, freq500_Orig,
                          freq500_AdvNat, freq500_AdvArt, std_orig, std_nat, std_art,
                          DATA_ID, ATCK_NAME[0])

    mean_freq_Orig = mf.combineMeanValueWithFreq(mean500_Orig, freq500_Orig)
    mean_freq_AdvNat = mf.combineMeanValueWithFreq(mean500_AdvNat, freq500_AdvNat)
    mean_freq_AdvArt = mf.combineMeanValueWithFreq(mean500_AdvArt, freq500_AdvArt)
    summary_boxplot=[]
    summary_boxplot.append(mean_freq_Orig)
    summary_boxplot.append(mean_freq_AdvNat)
    summary_boxplot.append(mean_freq_AdvArt)
    aux.saveBoxPlot(summary_boxplot, "", DATA_ID, atck=ATCK_NAME[0])
    aux.saveBoxPlot(summary_boxplot, "", DATA_ID, violin=True, atck=ATCK_NAME[0])
else:
    aux.saveVariable(valoresMetricas, "vectoresMetricas_%s.pkl" % (DATA_ID))