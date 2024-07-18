import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# Define the directory containing the text files and the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "data")
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")

load_dotenv()

print(f"Data directory: {data_dir}")
print(f"Persistent directory: {persistent_directory}")

# Check if the Chroma vector store already exists
if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")

    # Ensure the data directory exists
    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"The directory {data_dir} does not exist. Please check the path."
        )

    # List all user directories in the data directory
    user_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    print(f"User directories: {user_dirs}")

    # Read the text content from each user's files and store it with metadata
    documents = []
    for user_dir in user_dirs:
        user_path = os.path.join(data_dir, user_dir)
        user_files = [f for f in os.listdir(user_path) if f.endswith(".txt")]

        for user_file in user_files:
            file_path = os.path.join(user_path, user_file)
            loader = TextLoader(file_path)
            user_docs = loader.load()
            for doc in user_docs:
                # Add metadata to each document indicating its source and user
                print("This is the user dir and user file", user_dir, user_file)
                doc.metadata = {"source": user_file, "user": user_dir}
                print(f"Loaded document from {user_file} for user {user_dir}")
                documents.append(doc)

    # Split the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    # Display information about the split documents
    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")

    # Create embeddings
    print("\n--- Creating embeddings ---")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
    )  # Update to a valid embedding model if needed
    print("\n--- Finished creating embeddings ---")

    # Create the vector store and persist it
    print("\n--- Creating and persisting vector store ---")
    db = Chroma.from_documents(
        docs, embeddings, persist_directory=persistent_directory)
    print("\n--- Finished creating and persisting vector store ---")

else:
    print("Vector store already exists. No need to initialize.")