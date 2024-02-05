from astrapy.db import AstraDB # pip3 install torch torchvision
import doc_chunker
from dotenv import dotenv_values
import gen_embeddings
import json
from langchain_openai import OpenAI 
import os
import uuid

config = dotenv_values(".env")


# Initialize the client
db = AstraDB(
    token=config.get("ASTRA_DB_APPLICATION_TOKEN"),
    api_endpoint=config.get("ASTRA_DB_API_ENDPOINT"),
)



print(f"Connected to Astra DB: {db.get_collections()}")

llm = OpenAI(openai_api_key=config.get("OPENAI_API_KEY"), temperature=0)

# Fetching necessary environment variables for AstraDB configuration
ASTRA_DB_APPLICATION_TOKEN = config.get("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_API_ENDPOINT = config.get("ASTRA_DB_API_ENDPOINT")
COLLECTION_NAME = "strava_activities_from_code"


# Function to split the documents into batches of 20
def batch(documents, batch_size):
    for i in range(0, len(documents), batch_size):
        yield documents[i : i + batch_size]

# Chunk the sample file into paragraphs
paragraphs = doc_chunker.chunk_file("activities.csv")

# Create embeddings for each paragraph
embeddings = gen_embeddings.create_embeddings(paragraphs)
# Generating embeddings for each query using a custom embedding creation function
# embeddings = gen_embeddings.create_embeddings(queries)
print(f"embeddings {embeddings}")

documents = []  # Initialize an empty list to hold document dictionaries

for index, paragraph in enumerate(paragraphs):
    # Create a dictionary for the current document
    document = {
        "_id": str(uuid.uuid4()),  # Generate a unique ID for each document
        "text": paragraph,  # The text of the paragraph
        "vector": embeddings[index].tolist(),  # The corresponding embedding vector $vector
    }

    # Append the document dictionary to the list
    documents.append(document)

# Create collection
db.create_collection(collection_name=COLLECTION_NAME, dimension=768)

# Define the db collection
collection = db.collection(collection_name=COLLECTION_NAME)

# Insert the documents in batches
for batch_documents in batch(documents, 20):
    # Convert the batch of documents to a JSON array string
    json_array = json.dumps(batch_documents, indent=4)

    # Insert the batch of documents into the collection
    res = collection.insert_many(documents=batch_documents)
    print(f"Inserted batch with response: {res}")
queries = [
    "What was the longest run?",
    "What was the longest bike ride?",
    "When was the longest run?"
]

# Generating embeddings for each query using a custom embedding creation function
q_embeddings = gen_embeddings.create_embeddings(queries)

# Iterating through each query to perform a similarity search in the database
for index, query in enumerate(queries):
    # Converting the embedding to a list for the query
    embedding = q_embeddings[index].tolist()

    # Defining the sorting, options, and projection for the database query
    sort = {"$vector": embedding}
    options = {"limit": 2}
    projection = {"similarity": 1, "text": 1}

    # Executing the find operation on the collection with the specified parameters
    document_list = collection.find(sort=sort, options=options, projection=projection)

    print(query)
    # Iterating through the retrieved documents to print their content
    for document in document_list["data"]["documents"]:
        print(document["text"])
        print(document)
        # print(document["similarity"])
        print("\n\n")
