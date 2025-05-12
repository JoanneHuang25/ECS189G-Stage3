'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class Evaluate_Accuracy(evaluate):
    data = None
    
    def evaluate(self):
        print('evaluating performance...')
        true_y = self.data['true_y']
        pred_y = self.data['pred_y']
        
        # Calculate multiple evaluation metrics
        accuracy = accuracy_score(true_y, pred_y)
        
        # Precision
        precision_weighted = precision_score(true_y, pred_y, average='weighted')
        precision_macro = precision_score(true_y, pred_y, average='macro')
        
        # Recall
        recall_weighted = recall_score(true_y, pred_y, average='weighted')
        recall_macro = recall_score(true_y, pred_y, average='macro')
        
        # F1-score
        f1_weighted = f1_score(true_y, pred_y, average='weighted')
        f1_macro = f1_score(true_y, pred_y, average='macro')
        
        # Print detailed metrics
        print(f'Accuracy: {accuracy:.4f}')
        print(f'Precision (Weighted): {precision_weighted:.4f}, (Macro): {precision_macro:.4f}')
        print(f'Recall (Weighted): {recall_weighted:.4f}, (Macro): {recall_macro:.4f}')
        print(f'F1 Score (Weighted): {f1_weighted:.4f}, (Macro): {f1_macro:.4f}')
        
        # Return all metrics in a dictionary for further analysis/storage
        metrics = {
            'accuracy': accuracy,
            'precision_weighted': precision_weighted,
            'precision_macro': precision_macro,
            'recall_weighted': recall_weighted,
            'recall_macro': recall_macro,
            'f1_weighted': f1_weighted,
            'f1_macro': f1_macro
        }
        
        return metrics
        