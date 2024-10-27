from fastapi import FastAPI
from numpy import dot
from numpy.linalg import norm
from elasticsearch import Elasticsearch
from elasticsearch import AsyncElasticsearch
import torch
import torchvision.transforms as transforms
from torchvision import models
import numpy
import cv2
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel

from Text_Functions import get_embedding
from Image_Functions import *

app = FastAPI()

class TextQueryRequest(BaseModel):
    input: str

# Connect to Elasticsearch
es = AsyncElasticsearch(['http://localhost:9200'])
index_name = "ir_project"

@app.on_event("startup")
async def startup_event():
    # Optionally, you can check if the Elasticsearch server is up and running
    try:
        await es.ping()
        print("Elasticsearch is up and running.")
    except Exception as e:
        print(f"Elasticsearch connection failed: {e}")

    if await es.indices.exists(index=index_name):
        print(f"Index '{index_name}' Exists")
    else:
        print(f"Index '{index_name}' does not exist.")

    try:
        # Load Sentence Transformer model
        text_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Loaded Sentence Transformer model")
    except Exception as e:
        print("Error loading Sentence Transformer")

    try:
        # Load Vision Transformer model
        image_model = models.vit_l_16(weights=models.ViT_L_16_Weights.IMAGENET1K_V1)
        print("Loaded Vision Transformer")
        image_model.classifier = torch.nn.Identity()  # Remove the classification layer
        image_model.eval()  # Set the model to evaluation mode
    except Exception as e:
        print("Error loading Vision Transformer")

    print("-------------------------------------------------")
    print("Server Running Properly")
    print("------------------------------------------------- \n")


@app.get("/Text_Query")
async def Text_Query(request: TextQueryRequest):
    print("Recieved input")
    # User query
    user_query = request.input
    query_embedding = get_embedding(user_query,text_model)

    # Search for documents
    search_query = {
        "script_score": {
            "query": {
                "match_all": {}
            },
            "script": {
                "source": "cosineSimilarity(params.query_vector, 'Text_embedding') + 1.0",  # Adding 1 to avoid negative values
                "params": {
                    "query_vector": query_embedding.tolist()  # Convert to list if necessary
                }
            }
        }
    }

    response = await es.search(index=index_name, body={"query": search_query})
    res=[]
    # Analyze results
    if response['hits']['total']['value'] > 0:
        for doc in response['hits']['hits']:
            doc_details={
                        "Document ID":doc['_id'],"Score":doc['_score'],
                         "Title":doc['_source']['Title'],
                         "URL":doc['_source']['URL'],
                         "Abstract":doc['_source']['Abstract']
            }
            res.append(doc_details)
            # print(f"Document ID: {doc['_id']}, Score: {doc['_score']}")
            # print(f"Title: {doc['_source']['Title']}")
            # print(f"URL: {doc['_source']['URL']}")
            # print(f"Abstract: {doc['_source']['Abstract']}")
            # print("\n")

        return {"Result":res}


    return {"Result":"No Valid document found"}
