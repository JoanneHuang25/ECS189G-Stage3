'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.dataset import dataset
import pickle
import numpy as np


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
    
    def load(self):
        """
        Load the dataset
        :return: a dictionary containing training and testing data
        """
        print('loading data...')
        
        # Load the pickle file
        with open(self.dataset_source_folder_path + self.dataset_source_file_name, 'rb') as f:
            loaded_data = pickle.load(f)
        
        # Process training data
        train_X = []
        train_y = []
        for instance in loaded_data['train']:
            # Extract the image matrix
            image = instance['image']
            
            # Convert to flat array for the model input
            if self.dataset_source_file_name == 'ORL':
                # For ORL, use only one channel (since R=G=B)
                image = image[:, :, 0]  # Take only the first channel
            
            # Normalize the pixel values to [0, 1]
            image = image / 255.0
            
            # Add to our lists
            train_X.append(image)
            train_y.append(instance['label'])
        
        # Process testing data
        test_X = []
        test_y = []
        for instance in loaded_data['test']:
            # Extract the image matrix
            image = instance['image']
            
            # Convert to flat array for the model input
            if self.dataset_source_file_name == 'ORL':
                # For ORL, use only one channel (since R=G=B)
                image = image[:, :, 0]  # Take only the first channel
            
            # Normalize the pixel values to [0, 1]
            image = image / 255.0
            
            # Add to our lists
            test_X.append(image)
            test_y.append(instance['label'])
        
        # Convert to numpy arrays
        train_X = np.array(train_X)
        train_y = np.array(train_y)
        test_X = np.array(test_X)
        test_y = np.array(test_y)
        
        print(f'Loaded {len(train_X)} training instances and {len(test_X)} testing instances')
        
        return {'train': {'X': train_X, 'y': train_y}, 
                'test': {'X': test_X, 'y': test_y}}