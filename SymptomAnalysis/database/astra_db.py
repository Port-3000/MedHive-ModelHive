
# Setup Vector Database with AstraDB
# This script converts the disease-symptom dataset into embeddings and stores them in AstraDB

import pandas as pd
import numpy as np
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from cassandra.query import BatchStatement
import json
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import uuid

# Load environment variables
load_dotenv()

# AstraDB connection parameters
ASTRA_DB_ID = os.getenv("ASTRA_DB_ID")
ASTRA_DB_REGION = os.getenv("ASTRA_DB_REGION")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_KEYSPACE = os.getenv("ASTRA_DB_KEYSPACE", "disease_diagnosis")

# Function to initialize AstraDB connection
def get_astra_connection():
    cloud_config = {
        'secure_connect_bundle': 'secure-connect-disease-diagnosis.zip'
    }
    
    auth_provider = PlainTextAuthProvider(
        'token', ASTRA_DB_APPLICATION_TOKEN
    )
    
    cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
    session = cluster.connect()
    
    return session

# Function to create necessary tables in AstraDB
def create_tables(session):
    # Create keyspace if it doesn't exist
    session.execute(f"""
        CREATE KEYSPACE IF NOT EXISTS {ASTRA_DB_KEYSPACE} 
        WITH replication = {{'class': 'SimpleStrategy', 'replication_factor': '3'}}
    """)
    
    # Create disease_vectors table
    session.execute(f"""
        CREATE TABLE IF NOT EXISTS {ASTRA_DB_KEYSPACE}.disease_vectors (
            id uuid PRIMARY KEY,
            disease text,
            symptoms_vector list<float>,
            symptoms_text text,
            symptoms_binary map<text, int>
        )
    """)
    
    # Create ANN index for vector search
    session.execute(f"""
        CREATE CUSTOM INDEX IF NOT EXISTS disease_vector_index 
        ON {ASTRA_DB_KEYSPACE}.disease_vectors (symptoms_vector) 
        USING 'StorageAttachedIndex'
        WITH OPTIONS = {{ 
            'similarity_function': 'cosine',
            'dimension': 768
        }}
    """)
    
    print("Tables and indexes created successfully")

# Function to prepare and encode data
def prepare_data(csv_file_path):
    # Load data
    df = pd.read_csv(csv_file_path)
    
    # Initialize sentence transformer model for encoding
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Prepare data for AstraDB
    disease_vectors = []
    
    # Process each row
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Encoding diseases"):
        disease = row['diseases']
        
        # Get symptoms that are present (value = 1)
        symptoms_binary = {}
        symptoms_present = []
        
        for col in df.columns[1:]:  # Skip the disease column
            value = int(row[col])
            symptoms_binary[col] = value
            if value == 1:
                symptoms_present.append(col)
        
        # Create a text representation of the symptoms
        symptoms_text = ", ".join(symptoms_present)
        
        # Generate vector embedding for the symptoms text
        if symptoms_text:
            embedding = model.encode(symptoms_text).tolist()
        else:
            # If no symptoms, create a zero vector
            embedding = [0.0] * 768
        
        # Create a record
        disease_vectors.append({
            "id": uuid.uuid4(),
            "disease": disease,
            "symptoms_vector": embedding,
            "symptoms_text": symptoms_text,
            "symptoms_binary": symptoms_binary
        })
    
    return disease_vectors

# Function to insert data into AstraDB
def insert_data(session, disease_vectors):
    # Prepare insertion query
    insert_query = f"""
        INSERT INTO {ASTRA_DB_KEYSPACE}.disease_vectors 
        (id, disease, symptoms_vector, symptoms_text, symptoms_binary)
        VALUES (?, ?, ?, ?, ?)
    """
    
    # Prepare statement
    prepared = session.prepare(insert_query)
    
    # Insert records in batches
    batch_size = 50
    total_batches = (len(disease_vectors) + batch_size - 1) // batch_size
    
    for i in tqdm(range(total_batches), desc="Inserting data into AstraDB"):
        batch = BatchStatement()
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(disease_vectors))
        
        for record in disease_vectors[start_idx:end_idx]:
            batch.add(
                prepared, 
                (
                    record["id"],
                    record["disease"],
                    record["symptoms_vector"],
                    record["symptoms_text"],
                    record["symptoms_binary"]
                )
            )
        
        session.execute(batch)
    
    print(f"Successfully inserted {len(disease_vectors)} records into AstraDB")

# Main execution
def main():
    csv_file_path = "processed_disease_data.csv"  # Output from the PySpark EDA notebook
    
    # Connect to AstraDB
    print("Connecting to AstraDB...")
    session = get_astra_connection()
    
    # Create necessary tables
    print("Creating tables...")
    create_tables(session)
    
    # Prepare and encode data
    print("Preparing and encoding data...")
    disease_vectors = prepare_data(csv_file_path)
    
    # Insert data into AstraDB
    print("Inserting data into AstraDB...")
    insert_data(session, disease_vectors)
    
    # Close the connection
    session.shutdown()
    print("Process completed successfully!")

if __name__ == "__main__":
    main()