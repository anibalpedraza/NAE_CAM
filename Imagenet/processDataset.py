from PIL import Image
from os.path import join as fullfile, splitext
from os import walk, makedirs
from tqdm import tqdm

def cropCenterImage(img):
    # Define the region size of the center to crop
    cropSize=229
    width, height = img.size
    left = (width - cropSize)/2
    top = (height - cropSize)/2
    right = (width + cropSize)/2
    bottom = (height + cropSize)/2
    # Crop the center of the image
    img = img.crop((left, top, right, bottom))
    return img

def cropDiatomObject(img):
    # Define the region size of the center to crop
    cropSize=120
    offset=60
    width, height = img.size
    left = (width - cropSize)/2
    top = (height - cropSize + offset)/2
    right = (width + cropSize)/2
    bottom = (height + cropSize + offset)/2
    # Crop the center of the image
    img = img.crop((left, top, right, bottom))
    return img

def resizeImage(img):
    size=(224,224)
    img = img.resize(size)
    return img


def getClassFromDirname(dirname,filename):
    return dirname.split('\\')[-1]

def getClassFromFilename(dirname,filename):
    return filename.split('_')[0]

def convertAndProcessImages(inputPath,outputPath,outputFormat='png',
                            processingFunction=None,
                            classNameFunction=getClassFromDirname):

    # Read all subdirectores and convert images in specified format
    #nImages=sum([len(files) for _, _, files in walk(inputPath)])
    nClasses=len([subdir for subdir, _, _ in walk(inputPath)])
    for subdir, _, files in tqdm(walk(inputPath),total=nClasses):
        for file in tqdm(files):
            # Get name and extension of file
            name, ext = splitext(file)
            # Build output file folder
            classDir = classNameFunction(subdir,name)
            outputFolder = fullfile(outputPath, classDir)
            # Create folder if it doesn't exist
            makedirs(outputFolder,exist_ok=True)
            # Process image and save
            img = Image.open(fullfile(subdir,file))
            if processingFunction is not None:
                img = processingFunction(img)
            img.save(fullfile(outputFolder,name+'.'+outputFormat))

    print('Conversion finished')

if __name__ == '__main__':

    # V1: Convert all images in DatasetMerge to PNG
    '''
    baseInput='D:\\HANS_Diatoms'
    inputPath=fullfile(baseInput,'DatasetMerge')
    baseOutput='D:\\Dataset_NAE_CAM'
    outputPath=fullfile(baseOutput,'DatasetMerge_PNG')
    '''
    # V2: Apply a processing function to all images in frames_raw
    '''
    basePath='C:\\Users\\Aniba\\Documents\\Code\\VISILAB\\Dataset_NAE_CAM'
    inputPath=fullfile(basePath,'frames_raw')
    outputPath=fullfile(basePath,'frames_raw_crop_2') # 'frames_raw_crop'
    '''
    '''
    # V3: Apply the processing to the YOLO diatoms dataset from Alberto
    basePath='C:\\Users\\Aniba\\Documents\\Code\\VISILAB\\dataset_lucia_di_yolo'
    inputPath=fullfile(basePath,'images')
    outputPath=fullfile(basePath,'dataset_processed')
    '''
    # V4: Apply the processing to the cyano dataset for Alberto
    basePath='C:\\Users\\Aniba\\Documents\\Code\\VISILAB\\Dataset_NAE_CAM_Cyano'
    inputPath=fullfile(basePath,'Cyano')
    outputPath=fullfile(basePath,'dataset_cyano_processed')


    convertAndProcessImages(inputPath,outputPath,
                            processingFunction=resizeImage,
                            classNameFunction=getClassFromDirname)
