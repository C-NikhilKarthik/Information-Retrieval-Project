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
from fastapi.middleware.cors import CORSMiddleware

from Text_Functions import get_embedding
from Image_Functions import *

app = FastAPI()

origins = [
    "http://localhost",           # for local development
    "http://localhost:3000",      # for frontend running on port 3000
    "https://yourdomain.com",     # production domain
    # add other origins if needed
]

# Adding CORS middleware to FastAPI app
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,            # Allows specified origins
    allow_credentials=True,           # Allows cookies
    allow_methods=["*"],              # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],              # Allows all headers
)
class QueryRequest(BaseModel):
    text_input: str
    image_input:str

# Connect to Elasticsearch
es = AsyncElasticsearch(['http://localhost:9200'])
index_name = "ir_project"
text_model=None
image_model=None

@app.on_event("startup")
async def startup_event():
    global text_model,image_model
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


@app.post("/Text_Query")
async def Text_Query(request:QueryRequest):
    print("Recieved input")
    if text_model is None:
        return {"Result":"No text model loaded"}
    # User query
    user_query = request.text_input
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
                        "doc_id":doc['_id'],"score":doc['_score'],
                         "title":doc['_source']['Title'],
                         "url":doc['_source']['URL'],
                         "abstract":doc['_source']['Abstract']
            }
            res.append(doc_details)

        return {"Result":res}


    return {"Result":"No Valid document found"}


@app.post("/Image_Query")
async def Image_Query(request:QueryRequest):
    print("Recieved input")
    if image_model is None:
        return {"Result":"No image model loaded"}

    # Generate the query embedding for your image
    image_path = request.image_input
    query_embedding = generate_embedding(image_path,image_model)
    # Elasticsearch query
    query = {
        "query": {
            "nested": {
                "path": "Image_embedding_list",
                "score_mode": "max",  # Take the maximum score across nested vectors
                "query": {
                    "script_score": {
                        "query": {
                            "match_all": {}
                        },
                        "script": {
                            "source": "cosineSimilarity(params.query_vector, 'Image_embedding_list.embedding') + 1.0",  # Cosine similarity computation
                            "params": {
                                "query_vector": query_embedding  # Pass the reference vector as a parameter
                            }
                        }
                    }
                }
            }
        },
    }

    response = await es.search(index=index_name, body=query)
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

        return {"Result":res}


    return {"Result":"No Valid document found"}


@app.post("/Combined_Query")
async def Combined_Query(request:QueryRequest):
    print("Recieved input")
    if image_model is None or text_model is None:
        return {"Result":"All models not loaded"}

    text_input = request.text_input
    text_embedding = get_embedding(text_input, text_model)
    print("Text Query embedding generated")

    image_path = request.image_input
    image_embedding = generate_embedding(image_path, image_model)
    print("Image Query embedding generated")

    text_query = {
        "script_score": {
            "query": {
                "match_all": {}
            },
            "script": {
                "source": "cosineSimilarity(params.query_vector, 'Text_embedding') + 1.0",
                # Adding 1 to avoid negative values
                "params": {
                    "query_vector": text_embedding  # Convert to list if necessary
                }
            }
        }
    }

    response_text = await es.search(index=index_name, body={"query": text_query})

    image_query = {
        "query": {
            "nested": {
                "path": "Image_embedding_list",
                "score_mode": "max",  # Take the maximum score across nested vectors
                "query": {
                    "script_score": {
                        "query": {
                            "match_all": {}
                        },
                        "script": {
                            "source": "cosineSimilarity(params.query_vector, 'Image_embedding_list.embedding') + 1.0",
                            # Cosine similarity computation
                            "params": {
                                "query_vector": image_embedding  # Pass the reference vector as a parameter
                            }
                        }
                    }
                }
            }
        },
    }

    response_image = await es.search(index=index_name, body=image_query)

    '''---------------------------- Combine Query results ----------------------------------------'''
    sett = set()
    mapping = {}

    # Analyze results

    for doc in response_text['hits']['hits']:
        if doc['_id'] not in sett:
            sett.add(doc['_id'])
            mapping[doc['_id']] = {'Score': doc['_score'],
                                   'Source': doc['_source'],
                                    "Title": doc['_source']['Title'],
                                    "URL": doc['_source']['URL'],
                                    "Abstract": doc['_source']['Abstract']}
        # print(f"Document ID: {doc['_id']}, Score: {doc['_score']}, Source: {doc['_source']}")

    for doc in response_image['hits']['hits']:
        if doc['_id'] not in sett:
            sett.add(doc['_id'])
            mapping[doc['_id']] = {'Score': doc['_score'],
                                   'Source': doc['_source'],
                                   "Title": doc['_source']['Title'],
                                   "URL": doc['_source']['URL'],
                                   "Abstract": doc['_source']['Abstract']}
        else:
            mapping[doc['_id']]['Score'] += doc['_score']

        # print(f"Document ID: {doc['_id']}, Score: {doc['_score']}, Source: {doc['_source']}")

    if len(mapping)==0:
        return {"Result": "No Valid document found"}

    Documents = []
    for i in mapping:
        Documents.append([i, mapping[i]['Score']])

    sorted_Documents = sorted(Documents, key=lambda x: x[1], reverse=True)

    res = []
    for i in range(10):
        doc_id = sorted_Documents[i][0]
        doc_details = {
            "Document ID": doc_id, "Score": mapping[doc_id]["Score"],
            "Title": mapping[doc_id]["Title"],
            "URL": mapping[doc_id]["URL"],
            "Abstract": mapping[doc_id]["Abstract"]
        }
        res.append(doc_details)

    return {"Result": res}


