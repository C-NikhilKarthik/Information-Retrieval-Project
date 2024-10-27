import torch
import numpy
def get_embedding(text,model):
    # Tokenize and convert to tensor
    embedding = model.encode(text, convert_to_tensor=True)
    return embedding.cpu().numpy()