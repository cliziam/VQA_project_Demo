import cv2 as cv
import os
import json
import h5py
from tqdm import tqdm
#from transformers import ViTFeatureExtractor, ViTModel
from PIL import Image
import numpy as np
from torchvision import transforms
import torchvision.models as models
import torch


train_mapping = 'train_dset_mapping.json'
val_mapping = 'eval_dset_mapping.json'
train_dset_path = "C://Users/Davide//Desktop//train2014-resized"
val_dset_path = "C://Users//davide//Desktop//file-vqa//val2014"
h5path = "C://users//davide//desktop//dataset.h5"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def SIFT_extractor(image_name, sift, n_features=150):
    img = cv.imread(image_name)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    #kp = sift.detect(gray,None)
    #img=cv.drawKeypoints(gray,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #cv.imwrite('sift_keypoints.jpg',img)
    kp, des = sift.detectAndCompute(gray,None)
    while des.shape[0] > n_features:
        des = np.delete(des, -1, axis=0) # cut columns in excess
    if des.shape[0] < n_features:
        des = np.pad(des, [(n_features - des.shape[0], 0), (0,0)]) # pad to reach a number of rows equal to n_features
    return des
    
def ViT_extractor(image_path, feature_extractor, model):
    img = Image.open(image_path)
    inputs = feature_extractor(img, return_tensors="pt")
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    return last_hidden_states
    
def VGG_extractor(image_path, feature_extractor, convert_tensor):
    #img = Image.open(image_path)
    #t = convert_tensor(img)
    return feature_extractor(convert_tensor(Image.open(image_path)).to(device))

def get_idx(image_name):
    f = open(train_mapping)
    img_id = image_name[15:-4]
    i = 0
    while img_id[i] == '0':
        i += 1
    img_id = img_id[i:]
    data = json.load(f)
    return data[img_id]
            
    
    
def main(extract_for_train=True, extractor='SIFT'):
    assert extractor in ['SIFT', 'ViT', 'VGG']
    if extract_for_train:
        dset_path = train_dset_path
        n_images = 82783
    else:
        dset_path = val_dset_path
        n_images = 40504
    descriptors = [0 for _ in range(n_images)]
    if extractor == 'ViT':
        feat_extr = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
        model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        for img in tqdm(os.listdir(dset_path)):
            filename = os.fsdecode(img)
            idx = get_idx(filename)
            ef = SIFT_extractor(dset_path + '/' + filename, feat_extr, model)
            descriptors[idx] = ef
    elif extractor == 'SIFT':
        sift = cv.xfeatures2d.SIFT_create(nfeatures=150)
        for img in tqdm(os.listdir(dset_path)):
            filename = os.fsdecode(img)
            idx = get_idx(filename)
            ef = SIFT_extractor(dset_path + '/' + filename, sift)
            descriptors[idx] = ef
    else:
        vgg16 = models.vgg16(pretrained=True)
        vgg16_fe = vgg16.features.to(device)
        for img in tqdm(os.listdir(dset_path)):
            filename = os.fsdecode(img)
            idx = get_idx(filename)
            #ef = VGG_extractor(dset_path + '//' + filename, vgg16_fe)
            #descriptors[idx] = ef.reshape(512,49).detach().numpy()
            descriptors[idx] = VGG_extractor(dset_path + '//' + filename, vgg16_fe, transforms.ToTensor()).reshape(512,49).cpu().detach().numpy()
    
    print("building the h5 file...")
    with h5py.File(h5path,"w") as hdf:
        hdf.create_dataset('image_features', data=np.array(descriptors))
    print("done")
    
    
main(extractor='VGG')