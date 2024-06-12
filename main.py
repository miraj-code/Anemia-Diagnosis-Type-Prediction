# Import the necessary libraries for the program
import pandas as pd
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
import os
from tqdm import tqdm

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
for i in range(len(diagnosis)): diagnosis_dict[diagnosis[i]] = i

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

print(f'\nTraining Model \n'
      f'--------------- \n')
# Train the model iterating through the number of epoches
for i in tqdm(range(epoches)):
    # Forward the training dataset through the layers in the model
    y_pred = model.forward(X_train)
    # Calculate the loss
    loss = criterion(y_pred, y_train) 

    # Back propagate the loss to further train the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f'Completed! \n')

# Evaluate the test data
print(f'Evaluating Model...\n'
      f'-------------------')
with torch.no_grad():
    y_eval = model.forward(X_test)
    test_loss = criterion(y_eval, y_test)

# Display the necessary informations, like total correct diagnosis, training loss and testing loss 
correct = 0
with torch.no_grad():
    for i, data in enumerate(X_test):
        y_val = model.forward(data)

        if y_val.argmax().item() == y_test[i]:
            correct += 1

print(f'Evaluation Completed! \n'
      f'\n ----------------------------------------------------------------------------------------------------- \n' 
      f' Total Correct Diagnosis: {correct}/{len((X_test))} | Training Loss: {loss} | Testing Loss: {test_loss} \n'
      f' ----------------------------------------------------------------------------------------------------- \n')

# Detect the type of anemia with a new unseen data
new_diagnosis = torch.tensor([5.3, 25.845, 77.511, 1.88076, 5.14094, 4.8, 15.1, 46.1526, 82, 31.4, 38.3, 70, 14.31251157, 0.26028])
print(f'Diagnosing...\n'
      f'-------------')
with torch.no_grad():
    diagnosed_anemia = model(new_diagnosis)
    result = [diagnosis_type for diagnosis_type, id in diagnosis_dict.items() if id == diagnosed_anemia.argmax().item()]

    print(f'Diagnosis Completed! The diagnosed anemia type is {result}.')



