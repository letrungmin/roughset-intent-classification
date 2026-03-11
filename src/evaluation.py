import numpy as np

class RoughSetEvaluator:
    def __init__(self):
        """
        Initialize the evaluator for Rough Set classification metrics.
        """
        pass

    def evaluate(self, predictions, true_labels):
        """
        Evaluate the model performance based on Rough Set Theory principles.
        
        Args:
            predictions: List of tuples (pred_labels_list, region_type) 
                         from the RoughSetClassifier.
            true_labels: List of ground truth labels (IDs).
        """
        total_samples = len(true_labels)
        
        certain_count = 0
        correct_certain_count = 0
        boundary_count = 0
        
        for i in range(total_samples):
            pred_labels, region = predictions[i]
            true_label = true_labels[i]
            
            if region == 'CERTAIN':
                certain_count += 1
                # Check if the true label exists within the Certain region's predictions
                if true_label in pred_labels:
                    correct_certain_count += 1
            elif region == 'BOUNDARY':
                boundary_count += 1
                
        # 1. Coverage: Proportion of data where the system provides a definitive decision
        coverage = certain_count / total_samples if total_samples > 0 else 0
        
        # 2. Certainty Accuracy (Safety Accuracy): Accuracy rate within the Certain regions.
        # This is a critical KPI for Rough Set models, aiming for high precision (>95%).
        certainty_acc = correct_certain_count / certain_count if certain_count > 0 else 0
        
        # 3. Boundary Rate: Proportion of data classified as ambiguous/uncertain.
        boundary_rate = boundary_count / total_samples if total_samples > 0 else 0
        
        # Print Evaluation Report
        print("\n" + "="*50)
        print("ROUGH SET MODEL EVALUATION REPORT")
        print("="*50)
        print(f"Total Test Samples        : {total_samples}")
        print(f"Certain Region Samples    : {certain_count} ({coverage:.2%})")
        print(f"Boundary Region Samples   : {boundary_count} ({boundary_rate:.2%})")
        print("-" * 50)
        print(f"Certainty Accuracy (Target): {certainty_acc:.2%}")
        print("="*50)
        
        return {
            'coverage': coverage,
            'certainty_accuracy': certainty_acc,
            'boundary_rate': boundary_rate
        }