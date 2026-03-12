import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import os

def generate_full_tsne_figure():
    print("--- 1. Loading Datasets ---")
    banking = load_dataset("banking77")
    test_texts_id = banking['test']['text']
    
    clinc = load_dataset("clinc_oos", "plus")
    test_texts_ood = [x['text'] for x in clinc['test']][:1000]

    print("--- 2. Generating Embeddings (This might take a moment) ---")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    X_test_id = embedder.encode(test_texts_id, show_progress_bar=True)
    X_test_ood = embedder.encode(test_texts_ood, show_progress_bar=True)

    print("--- 3. Computing t-SNE Projection ---")
    n_samples = 400
    X_viz = np.vstack((X_test_id[:n_samples], X_test_ood[:n_samples]))
    y_viz = ['In-Distribution (Banking)'] * n_samples + ['Out-of-Distribution (OOS)'] * n_samples

    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    X_tsne = tsne.fit_transform(X_viz)

    print("--- 4. Plotting and Saving ---")
    df = pd.DataFrame(X_tsne, columns=['TSNE-1', 'TSNE-2'])
    df['Category'] = y_viz

    plt.figure(figsize=(10, 7), dpi=300)
    sns.set_style("whitegrid")
    
    ax = sns.scatterplot(
        data=df, x='TSNE-1', y='TSNE-2', hue='Category', 
        style='Category', palette=['#1f77b4', '#d62728'], 
        s=100, alpha=0.7, edgecolor='w', linewidth=0.5
    )

    plt.title('t-SNE Projection of Intent Manifolds\n(BANKING77 vs. CLINC150 OOS)', fontsize=15, fontweight='bold')
    plt.xlabel('Dimension 1', fontsize=12)
    plt.ylabel('Dimension 2', fontsize=12)
    plt.legend(title='Sample Type', title_fontsize='12', fontsize='11', loc='best')
    
    plt.tight_layout()
    plt.savefig('tsne_manifold.pdf', bbox_inches='tight')
    plt.show()
    print("Success! 'tsne_manifold.pdf' has been created.")

if __name__ == "__main__":
    generate_full_tsne_figure()