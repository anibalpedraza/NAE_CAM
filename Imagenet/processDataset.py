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

def convertAndProcessImages(inputPath,outputPath,outputFormat,processingFunction=None):

    # Read all subdirectores and convert images in specified format
    #nImages=sum([len(files) for _, _, files in walk(inputPath)])
    nClasses=len([subdir for subdir, _, _ in walk(inputPath)])
    for subdir, _, files in tqdm(walk(inputPath),total=nClasses):
        for file in tqdm(files):
            # Build output file folder
            classDir = subdir.split('\\')[-1]
            outputFolder = fullfile(outputPath, classDir)
            # Create folder if it doesn't exist
            makedirs(outputFolder,exist_ok=True)
            # Get name and extension of file
            name, ext = splitext(file)
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
    outputFormat='png'
    '''
    # V2: Apply a processing function to all images in frames_raw
    basePath='C:\\Users\\Aniba\\Documents\\Code\\VISILAB\\Dataset_NAE_CAM'
    inputPath=fullfile(basePath,'frames_raw')
    outputPath=fullfile(basePath,'frames_raw_crop_2') # 'frames_raw_crop'
    outputFormat='png'

    convertAndProcessImages(inputPath,outputPath,outputFormat,
                            processingFunction=cropDiatomObject)
