import io

import torchvision.transforms as transforms
from PIL import Image

def transform_image(image_bytes):
    '''
    transform the image to the same parameters you trained on
    eg; if the input image for the network was 224, then resize to 224
    if the images were normalized to some value, normalize them here
    '''
    my_transforms = transforms.Compose([
        transforms.Resize(255),#RESIZE the image to the same size as the size you trained on
        transforms.CenterCrop(224), #take only the center part of the image
        transforms.ToTensor(), #convert PIL image to a tensor
        transforms.Normalize([0.485, 0.456, 0.406],[0.229,0.224,0.225]) #normalize based on Imagenet means
    ])
    #the input will be in the form of bytes, 
    #the 
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)
