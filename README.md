
## 📍 Overview

MIRAX is a multimodal information retrieval system designed to provide comprehensive responses to queries by integrating data from various sources, such as text and images. Unlike traditional systems that focus on a single modality, MIRAX leverages advanced techniques in NLP and computer vision to handle both text and image-based retrieval seamlessly. This platform enables efficient and accurate retrieval of information across multiple modalities, offering users a unified search experience.

---

## 🚀 Getting Started

**_Requirements_**

Ensure you have the following dependencies installed on your system:

- **Node**: `v20.15.0 or above`
- **npm**: `v10.7.0 or above`

## Elasticsearch Installation

To set up Elasticsearch for the project, follow these steps:

### Step 1: Download and Install Elasticsearch
1. Visit the official [Elasticsearch download page](https://www.elastic.co/downloads/elasticsearch).  
2. Choose the version compatible with your system and download it.

Alternatively, use the command below to download Elasticsearch via the terminal:  
```bash
curl -L -O https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.10.1-linux-x86_64.tar.gz
```

### Step 2: Extract the Elasticsearch Files

After downloading, extract the tar file:

```bash
tar -xvf elasticsearch-8.10.1-linux-x86_64.tar.gz
cd elasticsearch-8.10.1
```

### Step 3: Start the Elasticsearch Service

Run the Elasticsearch service:

```bash
./bin/elasticsearch
```

### Step 4: Verify Installation
Once the service starts, verify it's running by visiting:
http://localhost:9200

You should see a JSON response indicating that Elasticsearch is running.

### Step 5:

Storing files on elasticsearch
...

---

### ⚙️ Installation

1. Clone the MIRAX repository:

```sh
git clone https://github.com/C-NikhilKarthik/MIRAX
```

2. Change to the project directory:

```sh
cd MIRAX
```

3. Install the dependencies for backend:

```sh
cd backend
pip install -r requirements.txt
```

4. Install the dependencies for frontend:
```sh
cd client
npm install
```

### 🏃‍♂️ Running MIRAX

Use the following command to run the backend MIRAX:

```sh
cd backend/App
python app.py
```

To run frontend with live reloading:

```sh
cd client
npm run dev
```

The application will be accessible at http://localhost:3000.

---

## Challenges  
1. **Integration of Multimodal Data**:  
   - Combining and processing information from both text and image sources effectively.  
2. **Efficient Query Handling**:  
   - Addressing multimodal queries (text and image) while maintaining speed and accuracy.  
3. **Model Training and Optimization**:  
   - Developing or fine-tuning a unified model for multimodal retrieval without overloading computational resources.  
4. **Semantic Understanding**:  
   - Ensuring the system understands and retrieves contextually relevant data from diverse modalities.

---

## Tech Stack Used  
- **Languages/Frameworks**: Python, FastAPI  
- **Libraries**:  
  - **For Text Retrieval**: Elasticsearch, NLP Models (e.g., BERT)  
  - **For Image Retrieval**: OpenCV, TensorFlow, PyTorch  
- **Concepts**: Multi-task learning to handle text and image modalities  
- **Tools**: Pre-trained models for object detection and semantic search to accelerate development and ensure accuracy.
---

## 👏 Contributors

This project exists thanks to all the people who contribute.

<p align="left">
  <a href="https://github.com/C-NikhilKarthik/MIRAX/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=C-NikhilKarthik/MIRAX" />
  </a>
</p>

---
