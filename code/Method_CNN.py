'''
Concrete MethodModule class for a specific learning MethodModule: CNN
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.method import method
from code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import numpy as np


class Method_CNN(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 100
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-3
    # Size for mini batches
    batch_size = 64
    # Dataset type
    dataset_name = 'MNIST'

    # Define the CNN model architecture
    def __init__(self, mName, mDescription, dataset_name='MNIST'):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.dataset_name = dataset_name
        
        # Different architectures based on dataset type
        if dataset_name == 'MNIST':
            # For MNIST dataset (grayscale, 28x28)
            self.input_channels = 1
            self.input_height = 28
            self.input_width = 28
            self.num_classes = 10
            
            # CNN layers
            self.conv_layers = nn.Sequential(
                # First convolutional block
                nn.Conv2d(self.input_channels, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                # Second convolutional block
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            
            # Calculate the size after convolutions and pooling
            conv_output_height = self.input_height // 4
            conv_output_width = self.input_width // 4
            conv_output_size = conv_output_height * conv_output_width * 64
            
            # Fully connected layers
            self.fc_layers = nn.Sequential(
                nn.Linear(conv_output_size, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, self.num_classes)
            )
            
        elif dataset_name == 'ORL':
            # For ORL dataset (grayscale, 112x92)
            self.input_channels = 1
            self.input_height = 112
            self.input_width = 92
            self.num_classes = 40  # 40 different persons
            
            # CNN layers
            self.conv_layers = nn.Sequential(
                # First convolutional block
                nn.Conv2d(self.input_channels, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                # Second convolutional block
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                # Third convolutional block
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            
            # Calculate the size after convolutions and pooling
            conv_output_height = self.input_height // 8
            conv_output_width = self.input_width // 8
            conv_output_size = conv_output_height * conv_output_width * 128
            
            # Fully connected layers
            self.fc_layers = nn.Sequential(
                nn.Linear(conv_output_size, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, self.num_classes)
            )
            
        elif dataset_name == 'CIFAR':
            # For CIFAR-10 dataset (color, 32x32x3)
            self.input_channels = 3
            self.input_height = 32
            self.input_width = 32
            self.num_classes = 10
            
            # CNN layers
            self.conv_layers = nn.Sequential(
                # First convolutional block
                nn.Conv2d(self.input_channels, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                # Second convolutional block
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                # Third convolutional block
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            
            # Calculate the size after convolutions and pooling
            conv_output_height = self.input_height // 8
            conv_output_width = self.input_width // 8
            conv_output_size = conv_output_height * conv_output_width * 128
            
            # Fully connected layers
            self.fc_layers = nn.Sequential(
                nn.Linear(conv_output_size, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, self.num_classes)
            )

    # Forward propagation
    def forward(self, x):
        '''Forward propagation'''
        # Reshape the input if it's not already in the right format
        if len(x.shape) == 2:  # If x is [batch_size, flattened_features]
            # Reshape according to the dataset type
            if self.dataset_name == 'MNIST':
                x = x.reshape(-1, 1, self.input_height, self.input_width)
            elif self.dataset_name == 'ORL':
                x = x.reshape(-1, 1, self.input_height, self.input_width)
            elif self.dataset_name == 'CIFAR':
                x = x.reshape(-1, 3, self.input_height, self.input_width)
        elif len(x.shape) == 3:  # If x is [batch_size, height, width] for grayscale
            # Add channel dimension for grayscale images
            if self.dataset_name != 'CIFAR':  # Only for grayscale datasets
                x = x.unsqueeze(1)  # [batch_size, 1, height, width]
        elif len(x.shape) == 4:  # If x is already [batch_size, channels, height, width]
            # Make sure for CIFAR that channels are in correct dimension
            if self.dataset_name == 'CIFAR' and x.shape[1] != 3:
                # If channels are in the last dimension [batch_size, height, width, channels]
                if x.shape[3] == 3:
                    x = x.permute(0, 3, 1, 2)  # Move channels to dim 1
        
        # Apply convolutional layers
        x = self.conv_layers(x)
        
        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1)
        
        # Apply fully connected layers
        x = self.fc_layers(x)
        
        return x

    def train(self, X, y):
        # Check for CUDA availability
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(device)
        
        # Optimizer and loss function
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()
        
        # For training accuracy investigation purpose
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')
        
        # Add a list to record loss values
        loss_values = []
        accuracy_values = []
        
        # Convert data to tensors
        if self.dataset_name == 'ORL':
            # ORL dataset starts labels at 1, adjust to 0-based for PyTorch
            y = np.array(y) - 1
        
        X_tensor = torch.FloatTensor(np.array(X))
        y_tensor = torch.LongTensor(np.array(y))
        
        # Create dataset and dataloader for mini-batch training
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Training loop
        for epoch in range(self.max_epoch):
            epoch_loss = 0
            correct = 0
            total = 0
            
            # Mini-batch training
            for batch_X, batch_y in dataloader:
                # Move data to device
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                # Forward pass
                y_pred = self.forward(batch_X)
                
                # Calculate loss
                train_loss = loss_function(y_pred, batch_y)
                epoch_loss += train_loss.item()
                
                # Backpropagation and optimization
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                
                # Calculate accuracy
                _, predicted = torch.max(y_pred.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
            
            # Calculate average loss and accuracy for the epoch
            avg_loss = epoch_loss / len(dataloader)
            epoch_accuracy = correct / total
            
            # Record metrics
            loss_values.append(avg_loss)
            accuracy_values.append(epoch_accuracy)
            
            # Print progress
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Loss: {avg_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')
        
        # Return metrics for plotting
        return {'loss_values': loss_values, 'accuracy_values': accuracy_values}
    
    def test(self, X):
        # Check for CUDA availability
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(device)
        
        # Set the model to evaluation mode
        self.eval()
        
        # Convert data to tensor and move to device
        X_tensor = torch.FloatTensor(np.array(X)).to(device)
        
        # Forward pass with no gradient calculation
        with torch.no_grad():
            y_pred = self.forward(X_tensor)
            # Get the predicted class indices
            _, predicted = torch.max(y_pred.data, 1)
        
        # For ORL dataset, adjust predictions back to 1-based
        if self.dataset_name == 'ORL':
            predicted = predicted + 1
        
        # Return predictions
        return predicted.cpu()
    
    def run(self):
        print('method running...')
        print('--start training...')
        metrics = self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        
        print('--Evaluation on test data:')
        evaluator = Evaluate_Accuracy('testing evaluator', '')
        evaluator.data = {'true_y': self.data['test']['y'], 'pred_y': pred_y.numpy()}
        test_metrics = evaluator.evaluate()
        
        return {'pred_y': pred_y.numpy(), 
                'true_y': self.data['test']['y'], 
                'metrics': test_metrics, 
                'loss_values': metrics['loss_values'],
                'accuracy_values': metrics['accuracy_values']}
            