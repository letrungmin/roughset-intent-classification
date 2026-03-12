import numpy as np
import time
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_curve, auc
import warnings
from scipy.special import logsumexp

warnings.filterwarnings('ignore')

def compute_fpr95(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    return fpr[np.argmin(np.abs(tpr - 0.95))]

def run_experiment(seed, X_train, y_train, X_test_mixed, y_test_mixed):
    results = {}
    # Logistic Regression with specific seed
    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(X_train, y_train)
    
    # MSP
    probs = clf.predict_proba(X_test_mixed)
    results['MSP'] = np.max(probs, axis=1)
    
    # Energy
    logits = clf.decision_function(X_test_mixed)
    results['Energy'] = logsumexp(logits, axis=1) 
    
    # KNN & TRS
    knn = NearestNeighbors(n_neighbors=30, metric='cosine', n_jobs=-1)
    knn.fit(X_train)
    distances, indices = knn.kneighbors(X_test_mixed)
    
    results['Deep kNN'] = -1.0 * np.mean(distances, axis=1)
    results['TRS (Ours)'] = np.max(probs, axis=1) * np.exp(-1.0 * np.mean(distances, axis=1))
    
    return results

def main():
    print("--- 1. LOADING & EMBEDDING ---")
    banking = load_dataset("banking77")
    clinc = load_dataset("clinc_oos", "plus")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    X_train = embedder.encode(banking['train']['text'])
    X_test_id = embedder.encode(banking['test']['text'])
    X_test_ood = embedder.encode([x['text'] for x in clinc['test']][:1000])
    
    X_test_mixed = np.vstack((X_test_id, X_test_ood))
    y_test_mixed = np.array([1]*len(X_test_id) + [0]*len(X_test_ood))

    seeds = [42, 123, 7, 2024, 99]
    all_metrics = {m: {'auroc': [], 'fpr95': []} for m in ['MSP', 'Energy', 'Deep kNN', 'TRS (Ours)']}

    print(f"--- 2. RUNNING 5 TRIALS ---")
    for seed in seeds:
        trial_results = run_experiment(seed, X_train, np.array(banking['train']['label']), X_test_mixed, y_test_mixed)
        for method, scores in trial_results.items():
            fpr, tpr, _ = roc_curve(y_test_mixed, scores)
            all_metrics[method]['auroc'].append(auc(fpr, tpr))
            all_metrics[method]['fpr95'].append(compute_fpr95(y_test_mixed, scores))

    print("\n--- 3. FINAL STATISTICAL RESULTS (Mean ± Std) ---")
    for m in all_metrics:
        auroc_mean, auroc_std = np.mean(all_metrics[m]['auroc']), np.std(all_metrics[m]['auroc'])
        fpr_mean, fpr_std = np.mean(all_metrics[m]['fpr95']), np.std(all_metrics[m]['fpr95'])
        print(f"{m}: AUROC = {auroc_mean:.4f} ± {auroc_std:.4f} | FPR95 = {fpr_mean:.4f} ± {fpr_std:.4f}")

if __name__ == "__main__":
    main()