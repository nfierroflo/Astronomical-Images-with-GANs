#Imports necesarios

import numpy as np
from torch import nn
import torch
import time
import pickle as pk
from torch.utils.data import DataLoader
import os
from torch.utils.data import Dataset

#Setear variables de entorno para limitar uso a solo una GPU

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

#Variables para el entrenamiento

epochs = 30
lr = 5e-4
dropout_p = 0.5
batch_size = 64
criterion = nn.CrossEntropyLoss()
#Usar GPU?
GPU_use = True
#Evaluacion en train cada cuantas iteraciones
train_each = 100

#Directorio donde se encuentra el dataset
save_dir = 'C:/Users/nfier/Documents/Magister/cursos/EL7006 Redes Neuronales y Teoría de Información para el Aprendizaje/proyecto/'

#Modelo y clases necesarias para cargar los datos

class StampDataset(Dataset):
    def __init__(self, images,features,onehot):
        self.img = images
        self.features = features
        self.labels = onehot

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        return self.img[idx,:,:,:], self.features[idx,:],self.labels[idx]

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
            nn.Linear(2304,64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        self.normalization = nn.BatchNorm1d(26)

        self.linears = nn.Sequential(
            nn.Linear(90, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 5),
            nn.Softmax()
        )

    def forward(self, images,features):
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
        features = self.normalization(features)
        x_meta = torch.cat((x_mean, features), 1)
        x_final = self.linears(x_meta)
        return x_final

#Paso de entrenamiento

def train_step(x_images,x_features, y_batch, model, optimizer, criterion):
    # Predicción
    y_predicted = model(x_images.float(),x_features.float())

    # Cálculo de loss

    loss = criterion(y_predicted, y_batch.cuda().long())

    # Actualización de parámetros
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return y_predicted, loss

#Paso de validacion

def validation_step(val_loader, model, criterion, use_gpu):
    cumulative_loss = 0
    cumulative_predictions = 0
    data_count = 0

    for img_val,feat_val, y_val in val_loader:
        if use_gpu:
            img_val = img_val.cuda()
            feat_val = feat_val.cuda()
            y_val = y_val.cuda().long()

        y_predicted = model(img_val.float(),feat_val.float())

        loss = criterion(y_predicted, y_val)

        class_prediction = torch.argmax(y_predicted, axis=1).long()

        cumulative_predictions += (y_val == class_prediction).sum().item()
        cumulative_loss += loss.item()
        data_count += y_val.shape[0]

    val_acc = cumulative_predictions / data_count
    val_loss = cumulative_loss / len(val_loader)

    return val_acc, val_loss


#Entrenamiento del modelo
def train_model(
    model,
    train_dataset,
    val_dataset,
    epochs,
    criterion,
    batch_size,
    lr,
    n_evaluations=train_each,
    use_gpu=False
):

    if use_gpu:
        model.cuda()

    # Definición de dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=use_gpu)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False, pin_memory=use_gpu)

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

    for epoch in range(epochs):
        print(f"\rEpoch {epoch + 1}/{epochs}")
        cumulative_train_loss = 0
        cumulative_train_corrects = 0
        train_loss_count = 0
        train_acc_count = 0

        # Entrenamiento del modelo
        for i, (images_batch,features_batch, y_batch) in enumerate(train_loader):
            model.train()
            if use_gpu:
                images_batch = images_batch.cuda()
                features_batch =  features_batch.cuda()
                y_batch = y_batch.cuda()

            y_predicted, loss = train_step(images_batch,features_batch, y_batch, model, optimizer, criterion)

            cumulative_train_loss += loss.item()
            train_loss_count += 1
            train_acc_count += y_batch.shape[0]

            # Numero de predicciones correctas
            class_prediction = torch.argmax(y_predicted, axis=1).long()
            cumulative_train_corrects += (y_batch == class_prediction).sum().item()

            if (iteration%n_evaluations==0):
                train_loss = cumulative_train_loss / train_loss_count
                train_acc = cumulative_train_corrects / train_acc_count

                print(f"Iteration {iteration} - Batch {i}/{len(train_loader)} - Train loss: {train_loss}, Train acc: {train_acc}")

            iteration += 1

        model.eval()
        with torch.no_grad():
            val_acc, val_loss = validation_step(val_loader, model, criterion, use_gpu)

        print(f"Val loss: {val_loss}, Val acc: {val_acc}")

        train_loss = cumulative_train_loss / train_loss_count
        train_acc = cumulative_train_corrects / train_acc_count

        curves["train_acc"].append(train_acc)
        curves["val_acc"].append(val_acc)
        curves["train_loss"].append(train_loss)
        curves["val_loss"].append(val_loss)

    print(f"Tiempo total de entrenamiento: {time.perf_counter() - t0:.4f} [s]")

    model.cpu()

    return curves

#Carga de datos
with open(save_dir + "stamp_dataset_21_new.pkl", "rb") as f:
    data = pk.load(f)


#Separacion de los datos
Train_dict = data['Train']
Validation_dict = data['Validation']
train_images = Train_dict['images']
validation_images = Validation_dict['images']
labels_train = Train_dict['class']
labels_val = Validation_dict['class']
features_train = Train_dict['features']
features_val = Validation_dict['features']


#Carga del dataset
train_dataset = StampDataset(np.transpose(train_images,[0,3,1,2]),features_train,labels_train)
val_dataset = StampDataset(np.transpose(validation_images,[0,3,1,2]),features_val,labels_val)

#Instancia del modelo
model = StampClassifier(dropout_p)

#Entrenamiento
curves = train_model(
    model,
    train_dataset,
    val_dataset,
    epochs,
    criterion,
    batch_size,
    lr,
    n_evaluations=train_each,
    use_gpu=GPU_use
)