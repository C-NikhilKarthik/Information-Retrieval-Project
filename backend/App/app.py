from flask import Flask, request, jsonify
from elasticsearch import Elasticsearch
import torch
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
import cv2
import base64
from sentence_transformers import SentenceTransformer
from Text_Functions import get_embedding
from Image_Functions import *
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=["http://localhost", "http://localhost:3000", "https://yourdomain.com"])

# Connect to Elasticsearch
es = Elasticsearch(['http://localhost:9200'])
index_name = "ir_project"
text_model = None
image_model = None

def load_models():
    global text_model, image_model
    try:
        text_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Loaded Sentence Transformer model")
    except Exception as e:
        print("Error loading Sentence Transformer:", e)

    try:
        image_model = models.vit_l_16(weights=models.ViT_L_16_Weights.IMAGENET1K_V1)
        image_model.classifier = torch.nn.Identity()
        image_model.eval()
        print("Loaded Vision Transformer")
    except Exception as e:
        print("Error loading Vision Transformer:", e)

load_models()


# Text query route
@app.route("/Text_Query", methods=["POST"])
def text_query():
    if text_model is None:
        return jsonify({"Result": "No text model loaded"})

    user_query = request.json["text_input"]
    query_embedding = get_embedding(user_query, text_model)

    search_query = {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.query_vector, 'Text_embedding') + 1.0",
                "params": {"query_vector": query_embedding.tolist()}
            }
        }
    }

    response = es.search(index=index_name, body={"query": search_query})
    results = [
        {
            "doc_id": doc['_id'],
            "score": doc['_score'],
            "title": doc['_source']['Title'],
            "url": doc['_source']['URL'],
            "abstract": doc['_source']['Abstract']
        }
        for doc in response['hits']['hits']
    ]
    return jsonify({"Result": results})


# Image query route
@app.route("/Image_Query", methods=["POST"])
def image_query():
    if image_model is None:
        return jsonify({"Result": "No image model loaded"})

    # print(request)
    image = request.json['image_input']
    image_embedding = generate_embedding(image, image_model)

    query = {
        "query": {
            "nested": {
                "path": "Image_embedding_list",
                "score_mode": "max",
                "query": {
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": "cosineSimilarity(params.query_vector, 'Image_embedding_list.embedding') + 1.0",
                            "params": {"query_vector": image_embedding.tolist()}
                        }
                    }
                }
            }
        }
    }

    response = es.search(index=index_name, body=query)
    results = [
        {
            "doc_id": doc['_id'],
            "score": doc['_score'],
            "title": doc['_source']['Title'],
            "url": doc['_source']['URL'],
            "abstract": doc['_source']['Abstract']
        }
        for doc in response['hits']['hits']
    ]
    return jsonify({"Result": results})


# Combined query route
@app.route("/Combined_Query", methods=["POST"])
def combined_query():
    if text_model is None or image_model is None:
        return jsonify({"Result": "All models not loaded"})

    text_input = request.json["text_input"]
    text_embedding = get_embedding(text_input, text_model)

    # image_data = request.json["image_input"].split(",")[1]
    # image_bytes = base64.b64decode(image_data)
    # image_np = np.frombuffer(image_bytes, np.uint8)
    # image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    image = request.json['image_input']
    image_embedding = generate_embedding(image, image_model)

    # Elasticsearch queries
    text_query = {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.query_vector, 'Text_embedding') + 1.0",
                "params": {"query_vector": text_embedding.tolist()}
            }
        }
    }
    image_query = {
        "query": {
            "nested": {
                "path": "Image_embedding_list",
                "score_mode": "max",
                "query": {
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": "cosineSimilarity(params.query_vector, 'Image_embedding_list.embedding') + 1.0",
                            "params": {"query_vector": image_embedding.tolist()}
                        }
                    }
                }
            }
        }
    }

    response_text = es.search(index=index_name, body={"query": text_query})
    response_image = es.search(index=index_name, body=image_query)

    results = {doc['_id']: doc['_score'] for doc in response_text['hits']['hits']}
    for doc in response_image['hits']['hits']:
        if doc['_id'] in results:
            results[doc['_id']] += doc['_score']
        else:
            results[doc['_id']] = doc['_score']

    sorted_docs = sorted(results.items(), key=lambda x: x[1], reverse=True)[:10]
    output = [
        {
            "doc_id": doc_id,
            "score": score,
            "title": response_text['hits']['hits'][0]['_source']['Title'],
            "url": response_text['hits']['hits'][0]['_source']['URL'],
            "abstract": response_text['hits']['hits'][0]['_source']['Abstract']
        }
        for doc_id, score in sorted_docs
    ]
    return jsonify({"Result": output})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
    app.run(debug=True)
