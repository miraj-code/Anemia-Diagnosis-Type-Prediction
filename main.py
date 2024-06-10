import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

load_dotenv()
path = os.getenv('DATASET_PATH')
data = pd.read_csv(path)

diagnosis = sorted(list(set(data['Diagnosis'])))
diagnosis_dict = {}
for i in range(len(diagnosis)):
    diagnosis_dict[diagnosis[i]] = i

X = data.drop('Diagnosis', axis = 1)
y = data['Diagnosis']

for i in range(len(y)):
    for key, value in diagnosis_dict.items():
        if y[i] == key:
            y = y.replace(y[i], value)

X = X.values
y = y.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

y_train = torch.CharTensor(y_train)
y_test = torch.CharTensor(y_test)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

epoches = 20

for i in range(epoches):
    y_pred = model.forward(X_train)
    loss = criterion(y_pred, y_train) 

    print(f'Epoch {i} | Loss: {loss}')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()