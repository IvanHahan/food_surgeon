import os

import firebase_admin
from firebase_admin import credentials, db
from langchain.schema import Document
from langchain.vectorstores import Pinecone

from food_surgeon.config import FIREBASE_URL
from food_surgeon.pinecone_embeddings import PineconeEmbeddings

if not firebase_admin._apps:
    cred = credentials.Certificate(".creds/ivan_firebase.json")
    firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_URL})


def create_rag_database(
    documents,
    document_ids,
    index_name="dishes",
):
    embeddings = PineconeEmbeddings()
    document_objs = [
        Document(page_content=d, metadata={"id": i})
        for d, i in zip(documents, document_ids)
    ]
    vectorstore = Pinecone.from_documents(
        document_objs, embeddings, index_name=index_name
    )
    return vectorstore


def get_vector_db(index_name="dishes"):
    embeddings = PineconeEmbeddings()
    return Pinecone.from_existing_index(index_name=index_name, embedding=embeddings)


def get_firebase_db(collection='dishes'):
    return db.reference(collection)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv(".env")

    # Fetch dishes from Firebase
    collection = "dishes"
    dishes = get_firebase_db(collection).get()

    # Extract descriptions for RAG database
    documents = [
        dish["name"]
        + f"\nТип:{dish.get('type')}\n"
        + "\nІнгредієнти:\n"
        + dish["ingredients"]
        + "\nОпис:\n"
        + dish["description"]
        for dish in dishes.values()
    ]

    document_ids = list(dishes.keys())

    # Create RAG database
    pinecone_api_key = os.getenv("PINECONE_API_KEY", "YOUR_PINECONE_API_KEY")
    pinecone_env = os.getenv("PINECONE_ENV", "YOUR_PINECONE_ENV")
    store = create_rag_database(documents, document_ids, collection)

    # Example query
    results = store.similarity_search("What are some dishes?", k=3)
    for res in results:
        print(res.page_content)
