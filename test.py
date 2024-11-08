import argparse
import logging
import wandb
import os
import pandas as pd
import torch
import rasterio
from predict import predict_img
import matplotlib.pyplot as plt

def confusion_matrix_1(y_true, y_pred, class_names):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    wandb.init(project="segmentacao_matriz_de_confusao") 

    cm = wandb.plot.confusion_matrix(
        y_true=y_true, preds=y_pred, class_names=class_names
    )

    wandb.log({"conf_mat": cm})
    return cm
    
def confusion_matrix_2(y_true, y_pred):
    
    N = max(max(y_true), max(y_pred)) + 1
    y_true = torch.tensor(y_true, dtype=torch.long)
    y_pred = torch.tensor(y_pred, dtype=torch.long)
    
    cm = torch.sparse.LongTensor(
         torch.stack([y_true, y_pred]), 
         torch.ones_like(y_true, dtype=torch.long),
         torch.Size([N, N])).to_dense()
    
    wandb.log({"conf_mat": cm})
    
    return cm

def load_image_as_flat_array(image_path):
    with rasterio.open(image_path) as src:
        img_array = src.read(1)  # Leia a primeira banda para classes
    return img_array.flatten()

def get_args():
    parser = argparse.ArgumentParser(description='Confusion matrix from model')
    parser.add_argument('--predicted', '-i', metavar='PREDICTED', help='Path of predicted images', required=True)
    parser.add_argument('--ground_truth', '-g', metavar='REAL', help='Path of ground truth images', required=True)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    
    predicteds_path = args.predicted
    gt_path = args.ground_truth
    df = pd.read_csv('data/classes.csv', header = None)
   
    y_true = []
    y_pred = []
   
    classes = df.iloc[:,1]
    
    for img_name in sorted(os.listdir(gt_path)):
        if img_name.endswith(".tiff"):
            img_path = os.path.join(gt_path, img_name)
            y_true.extend(load_image_as_flat_array(img_path))  # Adicione à lista de y_true

    # Carregar imagens previstas
    for img_name in sorted(os.listdir(predicteds_path)):
        if img_name.endswith("_OUT.png"):
            img_path = os.path.join(predicteds_path, img_name)
            y_pred.extend(load_image_as_flat_array(img_path))
        
    confusion_matrix_1(y_true, y_pred, classes)
    
    cm = confusion_matrix_2(y_true, y_pred)  # Ajuste conforme o seu código para gerar cm
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar()
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()

