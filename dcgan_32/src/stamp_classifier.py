import numpy as np
from torch import nn
import torch
import time
import pickle as pk
from torch.utils.data import DataLoader
import os
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from src.utils.metrics import Logger, get_inception_score
import pandas as pd


#Setear variables de entorno para limitar uso a solo una GPU

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

#Variables para el entrenamiento

class StampClassifier(nn.Module):
    def __init__(
            self,
            dropout_p,
    ):
        super().__init__()

        self.zpad = nn.ZeroPad2d(3)

        self.features_extractor = nn.Sequential(

            # Bloque 1
            nn.Conv2d(3, 32, kernel_size=(4,4), padding="valid"),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(5,5), padding="same"),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Bloque 2
            nn.Conv2d(32, 64, kernel_size=(5,5), padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(5,5), padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(5,5), padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(3136,64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        self.normalization = nn.BatchNorm1d(26)

        self.linears = nn.Sequential(
            #nn.Linear(90, 64),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 5),
            nn.Softmax(dim=1)
        )

    def forward(self, images):
        x_0 = self.zpad(images)
        x_1 = torch.rot90(x_0,1,[2,3])
        x_2 = torch.rot90(x_0, 2,[2,3])
        x_3 = torch.rot90(x_0, 3,[2,3])
        x0_p = self.features_extractor(x_0)
        x1_p = self.features_extractor(x_1)
        x2_p = self.features_extractor(x_2)
        x3_p = self.features_extractor(x_3)
        x_conc = torch.stack((x0_p, x1_p, x2_p, x3_p), dim=0)
        x_mean = torch.mean(x_conc, dim=0)
        #features = self.normalization(features)
        #x_meta = torch.cat((x_mean, features), 1)
        #x_final = self.linears(x_meta)
        x_final = self.linears(x_mean)
        return x_final

#Paso de entrenamiento

def train_step(x_images, y_batch, model, optimizer, criterion):
    # Predicción
    y_predicted = model(x_images.float())

    # Cálculo de loss

    loss = criterion(y_predicted, y_batch.cuda().long())

    # Actualización de parámetros
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return y_predicted, loss

#Paso de validacion

def validation_step(val_loader, model, criterion, use_gpu,best=False):
    cumulative_loss = 0
    cumulative_predictions = 0
    data_count = 0

    y_true = []  # True class labels
    y_pred = []  # Predicted class labels

    for img_val, y_val in val_loader:
        if use_gpu:
            img_val = img_val.cuda()
            y_val = y_val.cuda().long()

        y_predicted = model(img_val.float())

        loss = criterion(y_predicted, y_val)

        class_prediction = torch.argmax(y_predicted, axis=1).long()

        cumulative_predictions += (y_val == class_prediction).sum().item()
        cumulative_loss += loss.item()
        data_count += y_val.shape[0]

        # Append true and predicted labels for later evaluation
        y_true.extend(y_val.cpu().numpy())
        y_pred.extend(class_prediction.cpu().numpy())

    val_acc = cumulative_predictions / data_count
    val_loss = cumulative_loss / len(val_loader)

    if True:
        
        # Compute the confusion matrix
        confusion = confusion_matrix(y_true, y_pred)
        classes = ('AGN',
                    'SN',
                    'VS',
                    'asteroid',
                    'bogus')

        df_cm = pd.DataFrame(confusion, index=[i for i in classes],
                         columns=[i for i in classes])
        # Display the confusion matrix using seaborn and matplotlib
        plt.figure(figsize=(8, 6))

    return val_acc, val_loss, sns.heatmap(df_cm , annot=True, fmt='d', cmap='Blues').get_figure()


#Entrenamiento del modelo
def train_model(
    model,
    train_loader,
    val_loader,
    epochs,
    criterion,
    batch_size,
    lr,
    n_evaluations=100,
    use_gpu=False,
    save_models="",
    tb_log=True,
    model_id=1,
    tb_log_dir="logs_cls"
):

    if tb_log:
        logger = Logger(f'./{tb_log_dir}/cls_{model_id}')
    
    if use_gpu:
        model.cuda()


    # Optimizador
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Listas para guardar curvas de entrenamiento
    curves = {
        "train_acc": [],
        "val_acc": [],
        "train_loss": [],
        "val_loss": [],
    }

    t0 = time.perf_counter()

    iteration = 0
    best_val_acc = 0.1
    for epoch in range(epochs):
        print(f"\rEpoch {epoch + 1}/{epochs}")
        cumulative_train_loss = 0
        cumulative_train_corrects = 0
        train_loss_count = 0
        train_acc_count = 0

        best_val_loss = 1e6

        global_step = 0
        # Entrenamiento del modelo
        for i, (images_batch, y_batch) in enumerate(train_loader):
            model.train()
            if use_gpu:
                images_batch = images_batch.cuda()
                y_batch = y_batch.cuda()

            y_predicted, loss = train_step(images_batch, y_batch, model, optimizer, criterion)

            cumulative_train_loss += loss.item()
            train_loss_count += 1
            train_acc_count += y_batch.shape[0]

            # Numero de predicciones correctas
            class_prediction = torch.argmax(y_predicted, axis=1).long()
            cumulative_train_corrects += (y_batch == class_prediction).sum().item()

            if (iteration%n_evaluations==0):
                train_loss = cumulative_train_loss / train_loss_count
                train_acc = cumulative_train_corrects / train_acc_count

                #print(f"Iteration {iteration} - Batch {i}/{len(train_loader)} - Train loss: {train_loss}, Train acc: {train_acc}")

            iteration += 1


        model.eval()
        with torch.no_grad():
            val_acc, val_loss, cm_fig = validation_step(val_loader, model, criterion, use_gpu)
                
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_loss = val_loss
                best_epoch = epoch
                val_acc, val_loss, cm_fig = validation_step(val_loader, model, criterion, use_gpu,best=True)
                torch.save(model.state_dict(), f'saved_models/{save_models}/best_model_epoch{epoch}_{time.time()}.pt')

        print(f"Val loss: {val_loss}, Val acc: {val_acc}")


        train_loss = cumulative_train_loss / train_loss_count
        train_acc = cumulative_train_corrects / train_acc_count

        info = {
                        'train_loss': train_loss, 
                        'train_acc': train_acc,
                        'val_loss': val_loss,
                        'val_acc': val_acc,
                    }
        if tb_log:
            for tag, value in info.items():
                logger.scalar_summary(tag, value, epoch)

            logger.writer.add_figure("confussion_matrix", cm_fig, epoch)

            #for tag, value in model.named_parameters():
            #            tag = tag.replace('.', '/')
            #            logger.histo_summary(tag, value.data.cpu().numpy(), epoch)
            #            logger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), epoch)

        curves["train_acc"].append(train_acc)
        curves["val_acc"].append(val_acc)
        curves["train_loss"].append(train_loss)
        curves["val_loss"].append(val_loss)

        global_step += 1

    if tb_log:
        logger.writer.close()
    print(f"Tiempo total de entrenamiento: {time.perf_counter() - t0:.4f} [s]")
    print(f"Best Val loss: {best_val_loss}, Best Val acc: {best_val_loss}")
    model.cpu()
    return curves




def trainer(train_loader, val_loader,batch_size=32, epochs=30, criterion=nn.CrossEntropyLoss(),
            dropout_p = 0.5, GPU_use = True, train_each = 100, lr=5e-4, dir_name="", n_generated=5000):
    #Instancia del modelo
    model = StampClassifier(dropout_p)
    #Entrenamiento
    curves = train_model(
        model,
        train_loader,
        val_loader,
        epochs,
        criterion,
        batch_size,
        lr,
        n_evaluations=train_each,
        use_gpu=GPU_use,
        save_models=dir_name,
        model_id=n_generated
    )


    return curves