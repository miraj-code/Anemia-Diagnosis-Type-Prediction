# Import the necessary libraries for the program
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
import os

import torch
import torch.nn as nn

from model import model

# Load the path of the dataset from the environment variable
load_dotenv()
path = os.getenv('DATASET_PATH')

# Read the csv dataset file to get the necessary features
data = pd.read_csv(path)

# Create a dictionary for the output layer to replace the Characters to Numbers. It's used for later to convert the array to tensors.
# The dictionary is in the format of {diagnosis: id} 
diagnosis = sorted(list(set(data['Diagnosis'])))
diagnosis_dict = {}
for i in range(len(diagnosis)):
    diagnosis_dict[diagnosis[i]] = i

# Separate the input layers (are input features - X) and the output layers (y)
X = data.drop('Diagnosis', axis = 1)
y = data['Diagnosis']

# Replace the rows of y (which are Char type) to Number type
for i in range(len(y)):
    for key, value in diagnosis_dict.items():
        if y[i] == key:
            y = y.replace(y[i], value)

# Convert X and y to array
X = X.values
y = y.values

# Split X and y to train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

# Convert the X and y arrays to Tensor for the model to process them correctly
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

# Set the Loss function and Optimizer function
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Set a value for the number of epoches
epoches = 15000

# Train the model iterating through the number of epoches
for i in range(epoches):
    # Forward the training dataset through the layers in the model
    y_pred = model.forward(X_train)
    # Calculate the loss
    loss = criterion(y_pred, y_train) 

    print(f'Epoch {i+1} | Training Loss: {loss}')

    # Back propagate the loss to further train the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Evaluate the test data
with torch.no_grad():
    y_eval = model.forward(X_test)
    test_loss = criterion(y_eval, y_test)

# Display the necessary informations, like total correct diagnosis, training loss and testing loss 
correct = 0
with torch.no_grad():
    for i, data in enumerate(X_test):
        y_val = model.forward(data)

        # print(f'{i + 1}  {str(y_val)} \t {y_test[i]} \t {y_val.argmax().item()}')

        if y_val.argmax().item() == y_test[i]:
            correct += 1

print(f'\n ----------------------------------------------------------------------------------------------------- \n' 
      f' Total Correct Diagnosis: {correct} | Training Loss: {loss} | Testing Loss: {test_loss} \n'
      f' ----------------------------------------------------------------------------------------------------- \n')

# Detect the type of anemia with a new unseen data
new_diagnosis = torch.tensor([5.3, 25.845, 77.511, 1.88076, 5.14094, 4.8, 15.1, 46.1526, 82, 31.4, 38.3, 70, 14.31251157, 0.26028])

with torch.no_grad():
    print(model(new_diagnosis))

print(diagnosis_dict)


