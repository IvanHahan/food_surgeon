import os

import firebase_admin
from firebase_admin import credentials, db
from langchain.vectorstores import Pinecone

from food_surgeon.config import FIREBASE_URL
from food_surgeon.pinecone_embeddings import PineconeEmbeddings


def fetch_dishes_from_firebase(credential_path, db_url, path):
    # Initialize Firebase app
    cred = credentials.Certificate(credential_path)
    firebase_admin.initialize_app(cred, {"databaseURL": db_url})
    ref = db.reference(path)
    return ref.get()


def create_rag_database(
    documents,
    document_ids,
    index_name="dishes",
):
    embeddings = PineconeEmbeddings()
    vectorstore = Pinecone.from_texts(
        documents, embeddings, ids=document_ids, index_name=index_name
    )
    return vectorstore


def get_db(index_name="dishes"):
    embeddings = PineconeEmbeddings()
    return Pinecone.from_existing_index(index_name=index_name, embedding=embeddings)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv(".env")

    # Fetch dishes from Firebase
    collection = "dishes"
    dishes = fetch_dishes_from_firebase(
        ".creds/ivan_firebase.json", FIREBASE_URL, collection
    )

    # Extract descriptions for RAG database
    documents = [
        dish["name"]
        + "\Інгредієнти:\n"
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
