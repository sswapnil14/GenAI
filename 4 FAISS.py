
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline


# ## Task 1: Prepare Your Dataset [15 minutes]
# # Loading a sample dataset
documents = [
    "Deep learning models are solving complex problems.",
    "Generative AI can create lifelike images and videos.",
    "AI models need optimization to reduce biases.",
    "Natural language processing enables better human-computer interaction.",
    "Computer vision algorithms can detect objects in real-time.",
    "Reinforcement learning helps agents learn optimal strategies.",
    "Transfer learning accelerates model training on new tasks.",
    "Attention mechanisms have revolutionized sequence modeling."
]
 
print(f"Dataset prepared with {len(documents)} documents")
print("Sample document:", documents[0])

# ## Task 2: Create Embeddings and Index [15 minutes]
# Transform documents into vector representations and build a FAISS index.
# 1. Generate embeddings for your documents
# 2. Create a FAISS index
# 3. Add the embeddings to the index
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
embeddings = model.encode(documents)

print(f"Generated embeddings shape: {embeddings.shape}")

# Create FAISS index
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings.astype('float32'))

print(f"FAISS index created with {index.ntotal} vectors")


# ## Task 3: Retrieve and Generate [30 minutes]
# Retrieve relevant information and integrate it with language model queries.
# 1. Perform a query and retrieve documents
# 2. Create an enhanced prompt with retrieved context
# 3. Generate a response using a language model
# 4. Compare the enhanced response with a baseline response

# Perform a query and retrieve documents
query = "How do AI models optimize data?"
query_embedding = model.encode([query]).astype('float32')

k = 2  # Number of nearest neighbors
distances, indices = index.search(query_embedding, k)

retrieved_text = " ".join([documents[i] for i in indices[0]])

print(f"Query: {query}")
print(f"Retrieved documents: {retrieved_text}")

# Create enhanced prompt
complete_prompt = f"Info: {retrieved_text}\\nQ: {query}\\nA:"

# Integrate with LLM
generator = pipeline('text-generation', model='distilgpt2')
response = generator(complete_prompt, max_length=100)

print("Enhanced Response:", response[0]['generated_text'])

# Compare with baseline (no retrieval)
baseline_prompt = f"Q: {query}\\nA:"
baseline_response = generator(baseline_prompt, max_length=100)

print("\\nBaseline Response:", baseline_response[0]['generated_text'])
print("\\nComparison: The enhanced response should be more informed and contextual.")
