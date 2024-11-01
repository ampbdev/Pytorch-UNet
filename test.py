import wandb
import os

from predict import predict

def test(best_model, data_test, ground_truth):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    y_pred = []
    
    for img in os.listdir(data_test):
        if img.endswith(".tiff"):
            y_pred.append( predict(best_model,
                        img,
                        device,
                        scale_factor=1,
                        out_threshold=0.5))
    
    wandb.init(project="segmentacao_matriz_de_confusao")
    
    y_pred = pred_mask.view(-1).cpu().numpy() 
    
    y_true = true_mask.view(-1).cpu().numpy() 

    cm = wandb.plot.confusion_matrix(
        y_true=y_true, preds=y_pred, class_names=model.class_names
    )

    wandb.log({"conf_mat": cm})
    
    return