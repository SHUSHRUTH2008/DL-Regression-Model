# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Regression problems involve predicting a continuous output variable based on input features. Traditional linear regression models often struggle with complex patterns in data. Neural networks, specifically feedforward neural networks, can capture these complex relationships by using multiple layers of neurons and activation functions. In this experiment, a neural network model is introduced with a single linear layer that learns the parameters weight and bias using gradient descent.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: Generate Dataset

Create input values  from 1 to 50 and add random noise to introduce variations in output values .

### STEP 2: Initialize the Neural Network Model

Define a simple linear regression model using torch.nn.Linear() and initialize weights and bias values randomly.

### STEP 3: Define Loss Function and Optimizer

Use Mean Squared Error (MSE) as the loss function and optimize using Stochastic Gradient Descent (SGD) with a learning rate of 0.001.

### STEP 4: Train the Model

Run the training process for 100 epochs, compute loss, update weights and bias using backpropagation.

### STEP 5: Plot the Loss Curve

Track the loss function values across epochs to visualize convergence.

### STEP 6: Visualize the Best-Fit Line

Plot the original dataset along with the learned linear model.

### STEP 7: Make Predictions

Use the trained model to predict  for a new input value .

## PROGRAM

### Name:SHUSHRUTH M
### Register Number:212225243002

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(71)
X = torch.linspace(1, 50, 50).reshape(-1, 1)
e = torch.randint(-8, 9, (50, 1), dtype=torch.float)
y = 2 * X + 1 + e

# Plot the original data
plt.scatter(X, y, color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Generated Data for Linear Regression')
plt.show()

# Define the Linear Model Class
class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)

def forward(self, x):
    return self.linear(x)

# Initialize the Model
torch.manual_seed(59)  # Ensure same initial weights
model = Model(1, 1)

# Re-bind the optimizer to the newly initialized model parameters
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# Print Initial Weights and Bias
initial_weight = model.linear.weight.item()
initial_bias = model.linear.bias.item()
print("\nName: ")
print("Register No: ")
print(f'Initial Weight: {initial_weight:.8f}, Initial Bias: {initial_bias:.8f}\n')

# Define Loss Function & Optimizer
criterion = nn.MSELoss()  # Named 'criterion' to match your training loop
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# Train the Model
epochs = 100
losses = []


class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)

# Initialize training parameters
epochs = 100
losses = []

# Train the Model and print loss, weight, and bias for EVERY epoch
for epoch in range(epochs):
    # 1. Forward pass to get predictions
    y_pred = model(X)

    # 2. Calculate the loss
    loss = criterion(y_pred, y)
    losses.append(loss.item())

    # 3. Backpropagation & optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print loss, weight, and bias for EVERY epoch
    print(f'epoch: {epoch+1:2}  loss: {loss.item():10.8f}  '
          f'weight: {model.linear.weight.item():10.8f}  '
          f'bias: {model.linear.bias.item():10.8f}')


    # Print loss, weight, and bias for EVERY epoch
    print(f'epoch: {epoch:2}  loss: {loss.item():10.8f}  '
      f'weight: {model.linear.weight.item():10.8f}  '
      f'bias: {model.linear.bias.item():10.8f}')

# Plot Loss Curve
plt.plot(range(epochs), losses, color='blue')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.title('Loss Curve')
plt.show()

# Final Weights & Bias
final_weight = model.linear.weight.item()
final_bias = model.linear.bias.item()
print("\nName: ")
print("Register No: ")
print(f'\nFinal Weight: {final_weight:.8f}, Final Bias: {final_bias:.8f}')


#  Best-Fit Line Calculation
x1 = torch.tensor([X.min().item(), X.max().item()]) # Find min and max values of X
y1 = x1 * final_weight + final_bias # Compute corresponding y-values using trained model


# Plot Original Data & Best-Fit Line
plt.scatter(X, y, label="Original Data")
plt.plot(x1, y1, 'r', label="Best-Fit Line")
plt.xlabel('x')
plt.ylabel('y')
plt.title('Trained Model: Best-Fit Line')
plt.legend()
plt.show()

# Prediction for x = 120
x_new = torch.tensor([[120.0]])  # New input as a tensor
y_new_pred = model(x_new).item()  # Predict using trained model
print("\nName: ")
print("Register No: ")
print(f"\nPrediction for x = 120: {y_new_pred:.8f}")
```

### Dataset Information
Include screenshot of the generated data

### OUTPUT

<img width="704" height="571" alt="image" src="https://github.com/user-attachments/assets/eef8e882-9e56-4d04-b4d7-1b81db163cd0" />

<img width="735" height="572" alt="image" src="https://github.com/user-attachments/assets/e8931253-55da-4942-a346-1d43b82ddc28" />

<img width="705" height="557" alt="image" src="https://github.com/user-attachments/assets/724d9fdc-c950-465c-8750-c6d05464d2ee" />

### New Sample Data Prediction

<img width="352" height="94" alt="image" src="https://github.com/user-attachments/assets/79dd40cc-3f9d-4eae-b341-0aaa3aff58c8" />

## RESULT
Thus, a neural network regression model was successfully developed and trained using Google colab.
