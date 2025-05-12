import sys
import os

# Get the absolute path to the project root
current_file_path = os.path.abspath(__file__)
script_dir = os.path.dirname(current_file_path)
project_root = os.path.abspath(os.path.join(script_dir, "../.."))

# Add the project root to Python path
sys.path.insert(0, project_root)

# Now try to import the required modules
try:
    from code.stage_3_code.Dataset_Loader import Dataset_Loader
    from code.stage_3_code.Method_CNN import Method_CNN
    from code.stage_3_code.Result_Saver import Result_Saver
    from code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    print("Import successful!")
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Project root: {project_root}")
    print(f"Directory contents: {os.listdir(project_root)}")
    # Check if code directory exists
    code_dir = os.path.join(project_root, "code")
    if os.path.exists(code_dir):
        print(f"'code' directory exists, contains: {os.listdir(code_dir)}")
        stage_3_dir = os.path.join(code_dir, "stage_3_code")
        if os.path.exists(stage_3_dir):
            print(f"'stage_3_code' directory exists, contains: {os.listdir(stage_3_dir)}")
        else:
            print("'stage_3_code' directory doesn't exist!")
    else:
        print("'code' directory doesn't exist!")
    sys.exit(1)

#---- CNN script for image classification on multiple datasets ----
def run_experiment(dataset_name):
    print(f"\n{'='*50}")
    print(f"Running experiment for {dataset_name} dataset")
    print(f"{'='*50}\n")
    
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------

    # ---- objection initialization section ---------------
    data_obj = Dataset_Loader(f'{dataset_name} Dataset Loader', f'Load {dataset_name} dataset')
    data_obj.dataset_source_folder_path = os.path.join(project_root, 'data/stage_3_data/')
    data_obj.dataset_source_file_name = dataset_name
    
    method_obj = Method_CNN(f'{dataset_name} CNN', f'CNN for {dataset_name} classification', dataset_name)
    
    # Set epoch count based on dataset size
    if dataset_name == 'MNIST' or dataset_name == 'CIFAR':
        method_obj.max_epoch = 30  # Larger datasets, fewer epochs
    else:
        method_obj.max_epoch = 100  # Smaller dataset, more epochs
    
    # Create result folder
    result_folder = os.path.join(project_root, f'result/stage_3_result/{dataset_name.lower()}')
    os.makedirs(result_folder, exist_ok=True)
    
    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = f'{result_folder}/'
    result_obj.result_destination_file_name = f'CNN_{dataset_name}_results'
    result_obj.fold_count = 0  # Default fold
    
    evaluate_obj = Evaluate_Accuracy('accuracy', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    print('Loading data...')
    data = data_obj.load()
    
    # Pass the loaded data to the method
    method_obj.data = data
    
    # Run the method to train and test the model
    print('Training and testing the model...')
    result = method_obj.run()
    
    # Save results
    result_obj.data = result
    result_obj.save()
    
    print('************ Overall Performance ************')
    metrics = result['metrics']
    for metric_name, metric_value in metrics.items():
        print(f'{metric_name}: {metric_value:.4f}')
    
    # Create and save the loss plot
    print('Creating training loss and accuracy plots...')
    plt.figure(figsize=(12, 5))
    
    # Plot loss curve
    plt.subplot(1, 2, 1)
    plt.plot(result['loss_values'])
    plt.title(f'{dataset_name} - Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # Plot accuracy curve
    plt.subplot(1, 2, 2)
    plt.plot(result['accuracy_values'])
    plt.title(f'{dataset_name} - Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(f'{result_folder}/learning_curve_{dataset_name}.png')
    plt.close()
    
    print('************ Finish ************')
    return result
    # ------------------------------------------------------


if __name__ == '__main__':
    print("Starting CNN image classification experiments")
    
    # Check if GPU is available
    if torch.cuda.is_available():
        print("GPU is available! Using CUDA.")
    else:
        print("GPU not available. Using CPU.")
    
    # Run experiments for all three datasets
    
    # 1. MNIST dataset (digits)
    mnist_result = run_experiment('MNIST')
    
    # 2. ORL dataset (faces)
    orl_result = run_experiment('ORL')
    
    # 3. CIFAR dataset (objects)
    cifar_result = run_experiment('CIFAR')
    
    print("\nAll experiments completed!\n")
    
    # Print summary of results
    print("\nResults Summary:")
    print("-" * 50)
    print(f"MNIST Test Accuracy: {mnist_result['metrics']['accuracy']:.4f}")
    print(f"ORL Test Accuracy: {orl_result['metrics']['accuracy']:.4f}")
    print(f"CIFAR Test Accuracy: {cifar_result['metrics']['accuracy']:.4f}")
    print("-" * 50)
    

    