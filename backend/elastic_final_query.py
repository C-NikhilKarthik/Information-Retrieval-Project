from numpy import dot
from numpy.linalg import norm
from elasticsearch import Elasticsearch
import torch
import torchvision.transforms as transforms
from torchvision import models
import numpy
import cv2
from sentence_transformers import SentenceTransformer


def get_embedding(text,model):
    # Tokenize and convert to tensor
    embedding = model.encode(text, convert_to_tensor=True)
    return embedding.cpu().numpy()


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  # Resize to match MobileNetV2 input
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Function to load and preprocess image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    image = transform(image).unsqueeze(0)  # Apply transforms and add batch dimension
    return image


# Function to get image embedding
def generate_embedding(image_path,model):
    image = preprocess_image(image_path)
    with torch.no_grad():  # Disable gradient computation
        embedding = model(image)
    return embedding.flatten().numpy()


'''---------------------------- Main Section ----------------------------------------'''
# Load Sentence Transformer model
text_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Loaded Sentence Transformer model")

image_model = models.vit_l_16(weights=models.ViT_L_16_Weights.IMAGENET1K_V1)
print("Loaded Vision Transformer")
image_model.classifier = torch.nn.Identity()  # Remove the classification layer
image_model.eval()  # Set the model to evaluation mode

# Connect to Elasticsearch
es = Elasticsearch(['http://localhost:9200'])
# Check if the connection is successful
if es.ping():
    print("Connected to Elasticsearch!")
else:
    print("Error connecting to Elasticsearch")

index_name = "ir_project"


'''---------------------------- Only Text Query ----------------------------------------'''
#
# # User query
# user_query = "mobilenet convoltin model for images"
# query_embedding = get_embedding(user_query,text_model)
#
# # Search for documents
# search_query = {
#     "script_score": {
#         "query": {
#             "match_all": {}
#         },
#         "script": {
#             "source": "cosineSimilarity(params.query_vector, 'Text_embedding') + 1.0",  # Adding 1 to avoid negative values
#             "params": {
#                 "query_vector": query_embedding.tolist()  # Convert to list if necessary
#             }
#         }
#     }
# }
#
# response = es.search(index=index_name, body={"query": search_query})
#
# # Analyze results
# if response['hits']['total']['value'] > 0:
#     for doc in response['hits']['hits']:
#         print(f"Document ID: {doc['_id']}, Score: {doc['_score']}")
#         print(f"Title: {doc['_source']['Title']}")
#         print(f"URL: {doc['_source']['URL']}")
#         print(f"Abstract: {doc['_source']['Abstract']}")
#         print("\n")
#
# else:
#     print("No relevant documents found.")


# '''---------------------------- Only Image Query ----------------------------------------'''
#
# # Generate the query embedding for your image
# image_path = r'D:\programming\IR_Project\Images\Yolo\img-1.png'
# query_embedding = generate_embedding(image_path,image_model)
# # Elasticsearch query
# query = {
#     "query": {
#         "nested": {
#             "path": "Image_embedding_list",
#             "score_mode": "max",  # Take the maximum score across nested vectors
#             "query": {
#                 "script_score": {
#                     "query": {
#                         "match_all": {}
#                     },
#                     "script": {
#                         "source": "cosineSimilarity(params.query_vector, 'Image_embedding_list.embedding') + 1.0",  # Cosine similarity computation
#                         "params": {
#                             "query_vector": query_embedding  # Pass the reference vector as a parameter
#                         }
#                     }
#                 }
#             }
#         }
#     },
# }
#
# response = es.search(index=index_name, body=query)
#
# # Analyze results
# if response['hits']['total']['value'] > 0:
#     for doc in response['hits']['hits']:
#         print(f"Document ID: {doc['_id']}, Score: {doc['_score']}")
#         print(f"Title: {doc['_source']['Title']}")
#         print(f"URL: {doc['_source']['URL']}")
#         print(f"Abstract: {doc['_source']['Abstract']}")
#         print("\n")
#
# else:
#     print("No relevant documents found.")


'''---------------------------- Combined Query (Image + Text) ----------------------------------------'''
text_input = "convltin attntion blk"
text_embedding = get_embedding(text_input,text_model)
print("Text Query embedding generated")

image_path = r'D:\programming\IR_Project\Files\CBAM Convolutional Block Attention Module\img-1.png'
image_embedding = generate_embedding(image_path,image_model)
print("Image Query embedding generated")

text_query = {
    "script_score": {
        "query": {
            "match_all": {}
        },
        "script": {
            "source": "cosineSimilarity(params.query_vector, 'Text_embedding') + 1.0",  # Adding 1 to avoid negative values
            "params": {
                "query_vector": text_embedding  # Convert to list if necessary
            }
        }
    }
}

response_text = es.search(index=index_name, body={"query":text_query})
print("Text Query complete")

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
                        "source": "cosineSimilarity(params.query_vector, 'Image_embedding_list.embedding') + 1.0",  # Cosine similarity computation
                        "params": {
                            "query_vector": image_embedding  # Pass the reference vector as a parameter
                        }
                    }
                }
            }
        }
    },
}

response_image = es.search(index=index_name, body=image_query)
print("Image Query completed")


'''---------------------------- Combine Query results ----------------------------------------'''
sett=set()
mapping={}

# Analyze results

for doc in response_text['hits']['hits']:
    if doc['_id'] not in sett:
        sett.add(doc['_id'])
        mapping[doc['_id']]={'Score':doc['_score'],'Source':doc['_source']}
    # print(f"Document ID: {doc['_id']}, Score: {doc['_score']}, Source: {doc['_source']}")


for doc in response_image['hits']['hits']:
    if doc['_id'] not in sett:
        sett.add(doc['_id'])
        mapping[doc['_id']] = {'Score': doc['_score'], 'Source': doc['_source']}
    else:
        mapping[doc['_id']]['Score']+=doc['_score']

    # print(f"Document ID: {doc['_id']}, Score: {doc['_score']}, Source: {doc['_source']}")

Documents=[]
for i in mapping:
    Documents.append([i,mapping[i]['Score']])

sorted_Documents = sorted(Documents, key=lambda x: x[1],reverse=True)

for i in range(10):
    print(f"Doc ID: {sorted_Documents[i][0]}, Score: {sorted_Documents[i][1]}")
