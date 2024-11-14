from elasticsearch import Elasticsearch
import torch
import torchvision.transforms as transforms
from torchvision import models
from sentence_transformers import SentenceTransformer
import numpy as np
import fitz  # PyMuPDF
import re
import os
import cv2

# Function to extract main text content from the PDF
def extract_main_content(pdf_url):
    # Open the PDF file
    doc = fitz.open(pdf_url)
    main_content = ""

    # Iterate through the pages to extract text
    for page in doc:
        text = page.get_text()
        # text = re.sub(r'\d+', '', text)
        text = re.sub(r'[\d+\-.,\n\r]+', '', text)
        main_content += text+'\n'

    # Close the document
    doc.close()

    return main_content.strip()

# Function to chunk text into smaller segments
def chunk_text(text, chunk_size=512):
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield ' '.join(words[i:i + chunk_size])

# Function to generate embeddings using SBERT
def generate_embeddings(chunks,model):

    embeddings = model.encode(chunks, convert_to_tensor=True)
    return embeddings.cpu().numpy()


def filter_main_content(full_text):
    # Remove everything including and after the word "References"
    # Convert the text to lowercase for case-insensitive matching
    cutoff_index = full_text.find("Conclusion")

    if cutoff_index != -1:
        # Return the text up to the cutoff index
        return full_text[:cutoff_index].strip()

    # If "References" is not found, return the full text
    return full_text.strip()

# Main function to extract text and generate average embedding
def process_pdf_and_generate_embedding(pdf_url,model):
    # Step 1: Extract the main content from the PDF
    main_content = extract_main_content(pdf_url)
    main_content=filter_main_content(main_content)
    # print(main_content)

    # Step 2: Chunk the text into segments suitable for SBERT
    chunks = list(chunk_text(main_content, chunk_size=512))

    # Step 3: Generate embeddings for each chunk
    embeddings = generate_embeddings(chunks,model)

    # Step 4: Average the embeddings to create a single document embedding
    average_embedding = np.mean(embeddings, axis=0)
    # print("shape of average embedding is ",average_embedding.shape)

    return average_embedding


def extract_txt_content(txt_path):
    """Extract title, URL, and body text from a .txt file."""
    with open(txt_path, 'r',encoding='utf-8') as f:
        lines = f.readlines()

    # Initialize variables
    title = ""
    url = ""
    body_text = ""

    # Check if there are enough lines
    if len(lines) >= 5:
        title = lines[0].strip()  # First line is the title
        url = lines[2].strip()  # The line after the gap line is the URL
        body_text = ''.join(line.strip() + "\n" for line in lines[4:])  # The rest is the body text

    return title, url, body_text


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  # Resize to match input
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Function to load and preprocess image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not open or find the image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    image = transform(image).unsqueeze(0)  # Apply transforms and add batch dimension
    return image


# Function to get image embedding
def get_image_embedding(image_path,image_model):
    image = preprocess_image(image_path)
    with torch.no_grad():  # Disable gradient computation
        embedding = image_model(image)
    return embedding.flatten().numpy()

# Index documents with image embeddings in Elasticsearch
def get_nested_image_embeddings(image_paths,image_model):
    image_embeddings = []
    for image_path in image_paths:
        embedding = get_image_embedding(image_path,image_model)
        image_embeddings.append({"embedding":embedding})
        # print(embedding.shape)

    return image_embeddings


def create_index():
    mapping = {
        "mappings": {
            "properties": {
                "Title": {"type": "text"},
                "URL": {"type": "text"},
                "Abstract": {"type": "text"},
                "Text_embedding": {
                    "type": "dense_vector",
                    "dims": 384
                },
                "Image_embedding_list": {
                    "type": "nested",  # Use nested to allow arrays of dense vectors
                    "properties": {
                        "embedding": {
                            "type": "dense_vector",
                            "dims": 1000  # Model output dimension
                        }
                    }
                }
            }
        }
    }

    # Create index (if needed)
    if not es.indices.exists(index=index_name):
        es.indices.create(index=index_name, body=mapping)
        print(f"Index {index_name} created")
    else:
        print(f"Index {index_name} already exists")


'''---------------------------- Main Section ----------------------------------------'''


# Load SBERT model
text_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Loaded Sentence transformer model")

image_model = models.vit_l_16(weights=models.ViT_L_16_Weights.IMAGENET1K_V1)
image_model.classifier = torch.nn.Identity()  # Remove the classification layer
image_model.eval()  # Set the model to evaluation mode
print("Loaded Vision transformer Model")

# Connect to Elasticsearch
es = Elasticsearch(['http://localhost:9200'])
# Check if the connection is successful
if es.ping():
    print("Connected to Elasticsearch!")
else:
    print("Error connecting to Elasticsearch")


index_name = "ir_project"

create_index()

main_folder = r"/Users/nikhilkarthik/Desktop/IIIT Dharwad/7th Sem/IR/Files"
if not os.path.exists(main_folder):
    print("Path does not exist. Please check the path.")
else:
    print("Path exists. Entering for loop now...\n")
print("\n")
for subdir, _, files in os.walk(main_folder):
    # print("entered")
    if subdir==main_folder:
        continue
    average_embedding = []
    txt_title, txt_url, abstract_txt = "", "", ""
    image_paths=[]
    nested_image_embeddings=[]

    for file in files:
        file_path = os.path.join(subdir, file)
        # print(f"file path : {file_path}")
        if file.endswith('.pdf'):

            average_embedding = process_pdf_and_generate_embedding(file_path,text_model)
            # print("Generated average embedding for ",file)

        elif file.endswith('.txt'):
            # print(f"file path : {file_path}")
            txt_title, txt_url, abstract_txt = extract_txt_content(file_path)
            # print("read text file")

        elif file.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):  # Add any other image formats if needed
            image_paths.append(file_path)
            # print("image path appended")

    if len(image_paths)>0:
        nested_image_embeddings=get_nested_image_embeddings(image_paths,image_model)
        # print("nested image embeddings generated of length = ",len(nested_image_embeddings))

    # else:
    #     print("No images found")

        # print(nested_image_embeddings)

    document = {
        "Title": txt_title,
        "URL": txt_url,
        "Abstract": abstract_txt,
        "Text_embedding": average_embedding,
        "Image_embedding_list":nested_image_embeddings
    }

    # print("Document: \n",document)


    # Index the document
    es.index(index=index_name, id=txt_title, body=document)

    print(subdir," document stored successfully \n")

print("All Documents successfully stored")


'''---------------------------- Delete Index (Optional) ----------------------------------------'''

#
# # Connect to Elasticsearch
# es = Elasticsearch(['http://localhost:9200'])
# # Check if the connection is successful
# if es.ping():
#     print("Connected to Elasticsearch!")
# else:
#     print("Error connecting to Elasticsearch")
#
# index_name="ir_project"
# if es.indices.exists(index=index_name):
#     es.indices.delete(index=index_name)
#     print(f"Index '{index_name}' has been deleted.")
# else:
#     print(f"Index '{index_name}' does not exist.")


