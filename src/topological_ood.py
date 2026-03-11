import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_curve, auc
import warnings
import os

warnings.filterwarnings('ignore')

def load_and_prepare_data():
    # Load ID data (BANKING77)
    banking = load_dataset("banking77")
    train_texts_id = banking['train']['text']
    train_labels_id = np.array(banking['train']['label'])
    test_texts_id = banking['test']['text']

    # Load OOD data (CLINC150)
    clinc = load_dataset("clinc_oos", "plus")
    test_texts_ood = [x['text'] for x in clinc['test']][:1000]

    return train_texts_id, train_labels_id, test_texts_id, test_texts_ood

def extract_embeddings(texts, model_name='all-MiniLM-L6-v2'):
    # Convert texts to dense vectors
    embedder = SentenceTransformer(model_name)
    return embedder.encode(texts, show_progress_bar=True)

def evaluate_baseline(X_train, y_train, X_test_mixed):
    # Train softmax baseline
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    # Get maximum probability as confidence score
    probs = model.predict_proba(X_test_mixed)
    scores = np.max(probs, axis=1)
    return scores

def evaluate_topological_rough_set(X_train, X_test_mixed, n_neighbors=5):
    # Fit KNN on ID data
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
    knn.fit(X_train)

    # Use negative mean distance as confidence score
    distances, _ = knn.kneighbors(X_test_mixed)
    scores = -np.mean(distances, axis=1)
    return scores

def plot_roc_curve(y_true, sm_scores, rs_scores, save_path):
    # Calculate ROC and AUC for Baseline
    fpr_sm, tpr_sm, _ = roc_curve(y_true, sm_scores)
    auc_sm = auc(fpr_sm, tpr_sm)

    # Calculate ROC and AUC for Rough Set
    fpr_rs, tpr_rs, _ = roc_curve(y_true, rs_scores)
    auc_rs = auc(fpr_rs, tpr_rs)

    # Plot configuration
    plt.figure(figsize=(8, 8), dpi=150)
    plt.plot(fpr_sm, tpr_sm, color='red', linestyle='--', linewidth=2, 
             label=f'Softmax Baseline (AUROC = {auc_sm:.4f})')
    plt.plot(fpr_rs, tpr_rs, color='blue', linestyle='-', linewidth=2, 
             label=f'Topological Rough Set (AUROC = {auc_rs:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle=':', linewidth=1.5)

    plt.title('Out-of-Distribution Detection Performance', fontsize=14, fontweight='bold')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.7)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")

def main():
    print("Loading data...")
    train_texts_id, train_labels_id, test_texts_id, test_texts_ood = load_and_prepare_data()

    print("Extracting embeddings...")
    X_train_id = extract_embeddings(train_texts_id)
    X_test_id = extract_embeddings(test_texts_id)
    X_test_ood = extract_embeddings(test_texts_ood)

    # Combine test sets (1 for ID, 0 for OOD)
    X_test_mixed = np.vstack((X_test_id, X_test_ood))
    y_test_mixed = np.array([1]*len(X_test_id) + [0]*len(X_test_ood))

    print("Evaluating Softmax Baseline...")
    sm_scores = evaluate_baseline(X_train_id, train_labels_id, X_test_mixed)

    print("Evaluating Topological Rough Set...")
    rs_scores = evaluate_topological_rough_set(X_train_id, X_test_mixed)

    print("Generating ROC curve...")
    plot_roc_curve(y_test_mixed, sm_scores, rs_scores, save_path='src/ood_detection_roc.png')
    
    print("Execution completed successfully.")

if __name__ == "__main__":
    main()