#from idDict import idDictionary
class Imagen:
    def __init__(self, imageName, imageData, imageSize, imageId, networkName):
        self.name = imageName
        self.data = imageData
        self.size = imageSize #Indica el tamaño de la imagen (Ej:(224, 224))
        self.id = imageId #Indica el ID (n--------) de la clase real del objeto
        self.networkModelName = networkName #Indica el nombre de la red neuronal con el que se hizo la prediccion de los datos
        self.idName = imageId #idDictionary[imageId] #Indica el nombre de la clase real del objeto
        self.heatmap = 0
        self.predictionId = ''
        self.predictionName = ''
        self.attackName = '' #Indica el metodo de ataque con el cual se ha generado la imagen
        self.epsilon = '' #Indica el epsilon del metodo de ataque con el cual se ha generado la imagen
        self.advNatural = False #Indica si es un adversario natural
        self.closerOriginalImageName = '' #Indica el nombre de imagen original mas cercana en el caso de que sea adv natural
        
    def copyImage(self):
        img = Imagen(self.name, self.data, self.size, self.id, self.networkModelName)
        return img

    def modifyData(self, imageData):
        self.data = imageData

    def addNetworkModelName(self, networkName) :
        self.networkModelName = networkName

    def addHeatmap(self, heatmapData):
        self.heatmap = heatmapData

    def addPrediction(self, predID):
        self.predictionId = predID
        self.predictionName = predID #idDictionary[predID]s

    def addAdversarialData(self, attackName, epsilon):
        self.attackName = attackName
        self.epsilon = epsilon

    def addAdvNatural(self, isAdvNatural):
        self.advNatural = isAdvNatural

    def addCloserOriginalImageName(self, closerOriginalImageName) :
        self.closerOriginalImageName = closerOriginalImageName