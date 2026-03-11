from sentence_transformers import SentenceTransformer
import numpy as np

class TextEmbedder:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize the embedding model.
        Using 'all-MiniLM-L6-v2' as it is lightweight, fast on CPU, 
        and produces high-quality 384-dimensional vectors.
        """
        print(f"Loading Sentence-Transformers model: {model_name}...")
        self.model = SentenceTransformer(model_name)

    def encode_texts(self, text_list):
        """
        Convert a list of text strings into a mathematical vector matrix.
        """
        print(f"Encoding {len(text_list)} sentences into vectors...")
        # encode() returns a numpy array with shape [num_sentences, vector_dimension]
        embeddings = self.model.encode(text_list, show_progress_bar=True)
        print(f"Success! Vector matrix shape: {embeddings.shape}")
        return embeddings