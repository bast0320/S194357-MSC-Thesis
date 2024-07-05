import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class FFNN(nn.Module):
    def __init__(self, input_size, hidden_size=10, num_hidden_layers=3):
        super(FFNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        
        # Input layer
        self.input_layer = nn.Linear(input_size, hidden_size)
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_hidden_layers - 1):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size, input_size)
        
        # Activation function
        self.activation = nn.ReLU()
    
    def forward(self, x):
        # Input layer
        x = x.double()
        x = self.input_layer(x)
        x = self.activation(x)
        
        # Hidden layers
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.activation(x)
        
        # Output layer
        x = self.output_layer(x)

        # here we would then pass our output into our one_step_quantile_prediction function
        # and then update the weights of the model.

        # new_basis_for X should somehow be updated with x here.
        # and actually we should also run the adaptive quantile prediction here, since we then can output the y_pred, and
        # calculate loss based on that.
        
        return x.double()


# Quantile loss function in python: https://shrmtmt.medium.com/quantile-loss-in-neural-networks-6ea215fcee99
from functions_for_TAQR import one_step_quantile_prediction
from functions_for_TAQR import one_step_quantile_prediction_torch



# BACKUP ONE LINERS
# new_basis_X = torch.cat((new_basis_X[:i], model(train_data_X[i,:]).clone().double().unsqueeze(0), new_basis_X[i+1:]), dim=0)

# Define quantile loss function
def quantile_loss(preds, target, quantile):
    assert 0 < quantile < 1, "Quantile should be in (0, 1) range"
    errors = target - preds
    loss = torch.max((quantile - 1) * errors, quantile * errors)
    return torch.abs(loss).mean() # or sum??




def train_ffnn(model2, batch_size, num_epochs, train_data_X, train_data_Y, quantile_for_loss = 0.5, print_weights=False):
    
    '''
    we here assume the inputs are tensor, X and Y.
    
    '''

    train_data_X = train_data_X.clone().detach().requires_grad_(True)
    train_data_Y = train_data_Y.clone().detach().requires_grad_(True)

    std_loss = nn.MSELoss()
    model = FFNN(input_size = 10, hidden_size = 10, num_hidden_layers = 3)
    model = model.double()
    model.train()
    new_basis_X = torch.zeros_like(train_data_X)
    losses = []
    optimizer = optim.Adam(model.parameters(), lr=0.01)  # Adam optimizer with learning rate 0.001
    n_init = int(0.5*len(train_data_X))
    n_full = int(0.7*len(train_data_X))
    y_pred, y_actual, BETA = one_step_quantile_prediction_torch(X_input = train_data_X , 
                                                                Y_input = train_data_Y , 
                                                                n_init = n_init, n_full = n_full, quantile = quantile_for_loss)
        
        
    # ---- Training loop
    assert len(y_pred) == len(y_actual), "y_pred and y_actual should have the same length"
    for epoch in range(num_epochs):
        new_basis_X = torch.zeros_like(train_data_X)

        for i in range(len(y_pred)):
            loss = quantile_loss(preds=y_pred[i], target=y_actual[i], quantile=torch.tensor(quantile_for_loss))
            optimizer.zero_grad()
            loss.backward(retain_graph=True)  # Add retain_graph=True to retain the graph for multiple backward passes
            optimizer.step()
            losses.append(loss.item())

            outputs = model(train_data_X[i,:])
            new_basis_X[i,:] = outputs.clone().detach().requires_grad_(True)
        y_pred, y_actual, BETA = one_step_quantile_prediction_torch(X_input = new_basis_X , 
                                                                    Y_input = train_data_Y , 
                                                                    n_init=n_init, n_full=n_full, 
                                                                    quantile=quantile_for_loss)
        
    
        # loss = quantile_loss(preds=y_pred, target=y_actual, quantile=torch.tensor(quantile_for_loss))
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        # losses.append(loss.item())
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, RMSE loss: {torch.norm(y_pred-y_actual)}")
    
    import matplotlib.pyplot as plt
    plt.plot(losses)

    return model


def train_ffnn_backup_newer(model, batch_size, num_epochs, train_data_X, train_data_Y, quantile_for_loss = 0.5, print_weights=False):
    
    # make data into tensor
    train_data_X = torch.tensor(train_data_X, dtype=torch.float32)
    train_data_Y = torch.tensor(train_data_Y, dtype=torch.float32)
    # quantile_for_loss = torch.tensor(quantile_for_loss)

    new_basis_X = train_data_X


    # Define the loss function and optimizer
    # criterion = nn.MSELoss()  # Mean Squared Error loss
    optimizer = optim.Adam(model.parameters(), lr=0.1)  # Adam optimizer with learning rate 0.001
    n_init = int(0.5*len(train_data_X))
    n_full = int(0.7*len(train_data_X))
    y_pred, y_actual, BETA = one_step_quantile_prediction(X_input = train_data_X.detach().numpy(), 
                                                                Y_input = train_data_Y.detach().numpy(), 
                                                                n_init = n_init, n_full = n_full, quantile = quantile_for_loss)
    # Training loop
    for epoch in range(num_epochs):



        # let's here run the entire simplex algo, with eventuelt opdatetes basis.
        for i in range(n_init, n_full):
            # replace new basis
            new_basis_X[i,:] = model(train_data_X[i,:])

        y_pred, y_actual, BETA = one_step_quantile_prediction(X_input = new_basis_X.detach().numpy(), 
                                                                Y_input = train_data_Y.detach().numpy(), 
                                                                n_init = n_init, n_full = n_full, quantile = quantile_for_loss)
                

        # print(y_pred, y_actual)
        # make sure y_pred is tensor
        y_pred = torch.tensor(y_pred, dtype=torch.float32, requires_grad=True)
        y_actual = torch.tensor(y_actual, dtype=torch.float32, requires_grad=True)

        # loss = criterion(outputs, batch_labels)
        # print("y_pred: ", y_pred, "target: ",  target_quantile)
        loss = quantile_loss(preds = y_pred, target = y_actual, quantile = torch.tensor(quantile_for_loss)) # for now we just do target = actual, since with the 0.5 quantile, this will be pretty close
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        # Print the loss for every epoch
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    
    return model


def train_ffnn_backup_10042024(model, batch_size, num_epochs, train_data_X, train_data_Y, quantile_for_loss = 0.5, print_weights=False):
    
    # make data into tensor
    train_data_X = torch.tensor(train_data_X, dtype=torch.float32)
    train_data_Y = torch.tensor(train_data_Y, dtype=torch.float32)
    # quantile_for_loss = torch.tensor(quantile_for_loss)


    # Define the loss function and optimizer
    # criterion = nn.MSELoss()  # Mean Squared Error loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer with learning rate 0.001
    
    # Training loop
    for epoch in range(num_epochs):
        for i in range(300, len(train_data_X)-3):
            # Get a batch of training data
            # batch_data = train_data_X[i:i+batch_size]
            # batch_labels = train_labels_y[i:i+batch_size]
            # find the quantile for the batch

            target_quantile = torch.quantile((train_data_Y[:i]), torch.tensor(quantile_for_loss) )
            
            # Forward pass
            outputs = model(train_data_X[i,:]) # take the current data and make it into a basis for the data. 
            # print("outputs: ", torch.norm(outputs), outputs.shape)
            
            # i -> i. # # the outputs, should replace the current line in the X matrix
            train_data_X[i,:] = outputs.clone() 
            n_full = i+3
            n_init = i
            print("epoch: ", epoch, "i: ", i, "n_full: ", n_full, "n_init: ", n_init)
            # Print the weights if print_weights is True
            if print_weights:
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        print(f"Layer: {name}, Weight: {param.data}")
            
            
            y_pred, y_actual, BETA = one_step_quantile_prediction(X_input = train_data_X.detach().numpy(), 
                                                                  Y_input = train_data_Y.detach().numpy(), 
                                                                  n_init = n_init, n_full = n_full, quantile = quantile_for_loss)

            print(y_pred, y_actual)
            # make sure y_pred is tensor
            y_pred = torch.tensor(y_pred, dtype=torch.float32, requires_grad=True)
            y_actual = torch.tensor(y_actual, dtype=torch.float32, requires_grad=True)

            # loss = criterion(outputs, batch_labels)
            # print("y_pred: ", y_pred, "target: ",  target_quantile)
            loss = quantile_loss(preds = y_pred, target = y_actual, quantile = torch.tensor(quantile_for_loss)) # for now we just do target = actual, since with the 0.5 quantile, this will be pretty close
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Print the loss for every epoch
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    
    return model


# # Example usage
# input_size = 5
# batch_size = 32
# num_epochs = 100 

# # Assuming you have your training data and labels
# train_data_X = torch.randn(100, input_size)  # Replace with your actual training data
# train_labels_y = torch.randn(100, input_size)  # Replace with your actual training labels

# # Train the model
# trained_model = train_ffnn(input_size, batch_size, num_epochs, train_data_X, train_labels_y)

# print(trained_model)