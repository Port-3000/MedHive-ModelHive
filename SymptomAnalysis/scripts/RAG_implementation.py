
# Disease Diagnosis LLM with RAG using Groq
# This module implements the Retrieval-Augmented Generation for disease diagnosis

import os
import json
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
import groq
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema import Document

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# AstraDB connection parameters
ASTRA_DB_ID = os.getenv("ASTRA_DB_ID")
ASTRA_DB_REGION = os.getenv("ASTRA_DB_REGION")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_KEYSPACE = os.getenv("ASTRA_DB_KEYSPACE", "disease_diagnosis")

# Groq API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Embedding model
MODEL_NAME = "all-MiniLM-L6-v2"

class DiseaseVectorStore:
    """Vector store for disease symptoms using AstraDB"""
    
    def __init__(self):
        self.session = self._connect_to_astra()
        self.encoder = SentenceTransformer(MODEL_NAME)
    
    def _connect_to_astra(self):
        """Connect to AstraDB"""
        cloud_config = {
            'secure_connect_bundle': 'secure-connect-disease-diagnosis.zip'
        }
        
        auth_provider = PlainTextAuthProvider(
            'token', ASTRA_DB_APPLICATION_TOKEN
        )
        
        cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
        session = cluster.connect()
        
        return session
    
    def similarity_search(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar disease vectors based on query text
        
        Args:
            query_text: Natural language description of symptoms
            top_k: Number of top results to return
            
        Returns:
            List of disease records with similarity scores
        """
        # Encode query text
        query_vector = self.encoder.encode(query_text).tolist()
        
        # Execute similarity search
        query = f"""
            SELECT disease, symptoms_text, symptoms_binary,
                   astra_vector_distance(symptoms_vector, {query_vector}) as similarity
            FROM {ASTRA_DB_KEYSPACE}.disease_vectors
            ORDER BY symptoms_vector ANN OF {query_vector} LIMIT {top_k}
        """
        
        rows = self.session.execute(query)
        
        results = []
        for row in rows:
            results.append({
                "disease": row.disease,
                "symptoms_text": row.symptoms_text,
                "symptoms_binary": row.symptoms_binary,
                "similarity": row.similarity
            })
        
        return results
    
    def close(self):
        """Close AstraDB connection"""
        if self.session:
            self.session.shutdown()


class SymptomExtractor:
    """Extract symptoms from natural language descriptions"""
    
    def __init__(self):
        """Initialize the symptom extractor with Groq LLM"""
        self.llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model_name="llama3-70b-8192",  # Using Llama 3 for better extraction
        )
        
        # Get list of all symptoms
        df = pd.read_csv("processed_disease_data.csv")
        self.all_symptoms = list(df.columns[1:])  # All symptom columns except disease
    
    def extract_symptoms(self, user_input: str) -> List[str]:
        """
        Extract symptoms from user natural language input
        
        Args:
            user_input: Natural language description of symptoms
            
        Returns:
            List of extracted symptoms from the predefined symptom set
        """
        # Create symptoms list as a multi-line string for better LLM processing
        symptoms_list = "\n".join(self.all_symptoms)
        
        # Construct prompt for symptom extraction
        prompt = f"""
        Below is a list of standardized medical symptoms:
        
        {symptoms_list}
        
        A patient has described their condition as follows:
        "{user_input}"
        
        Based on their description, identify which symptoms from the standardized list are present.
        Format your response as a JSON array of symptom names only.
        Only include symptoms that are explicitly mentioned or strongly implied in the patient's description.
        Do not include symptoms that are not mentioned.
        """
        
        try:
            response = self.llm.invoke(prompt)
            response_text = response.content
            
            # Extract JSON array from response
            import re
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                extracted_symptoms = json.loads(json_match.group())
            else:
                # Fallback if JSON not properly formatted
                extracted_symptoms = [s.strip() for s in response_text.split(',') if s.strip() in self.all_symptoms]
            
            return extracted_symptoms
        
        except Exception as e:
            logger.error(f"Error extracting symptoms: {e}")
            return []


class DiagnosisRAG:
    """Retrieval-Augmented Generation system for disease diagnosis"""
    
    def __init__(self):
        """Initialize the RAG system"""
        self.vector_store = DiseaseVectorStore()
        self.symptom_extractor = SymptomExtractor()
        self.llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model_name="mixtral-8x7b-32768",  # Using Mixtral for diagnosis
        )
    
    def _format_context(self, results: List[Dict[str, Any]]) -> str:
        """Format similar disease results as context"""
        context = []
        
        for i, result in enumerate(results, 1):
            context.append(f"Disease {i}: {result['disease']}")
            context.append(f"Symptoms: {result['symptoms_text']}")
            context.append(f"Similarity score: {result['similarity']:.4f}")
            context.append("")
        
        return "\n".join(context)
    
    def diagnose(self, user_input: str) -> Dict[str, Any]:
        """
        Diagnose diseases based on user input
        
        Args:
            user_input: Natural language description of symptoms
            
        Returns:
            Dict containing diagnosis results
        """
        # Step 1: Extract symptoms from user input
        extracted_symptoms = self.symptom_extractor.extract_symptoms(user_input)
        
        # Step 2: Prepare query text from extracted symptoms
        query_text = ", ".join(extracted_symptoms) if extracted_symptoms else user_input
        
        # Step 3: Retrieve similar disease vectors
        similar_diseases = self.vector_store.similarity_search(query_text, top_k=5)
        
        # Step 4: Format context for LLM
        context = self._format_context(similar_diseases)
        
        # Step 5: Prepare diagnosis prompt
        diagnosis_prompt = f"""
        You are a medical diagnostic assistant. Your task is to analyze the patient's symptoms and suggest possible diagnoses.
        
        PATIENT DESCRIPTION:
        {user_input}
        
        EXTRACTED SYMPTOMS:
        {', '.join(extracted_symptoms)}
        
        SIMILAR CASES FROM MEDICAL DATABASE:
        {context}
        
        Based on the patient's description and the similar cases, provide the following:
        1. Most likely diagnoses (list up to 3 possibilities in order of likelihood)
        2. Explanation for each diagnosis
        3. Important symptoms that support each diagnosis
        4. Any critical warning signs the patient should be aware of
        5. General recommendations (e.g., see a doctor, home care, etc.)
        
        IMPORTANT: Always include a disclaimer that this is not a replacement for professional medical advice.
        """
        
        # Step 6: Generate diagnosis
        response = self.llm.invoke(diagnosis_prompt)
        
        return {
            "user_input": user_input,
            "extracted_symptoms": extracted_symptoms,
            "similar_diseases": similar_diseases,
            "diagnosis": response.content
        }
    
    def close(self):
        """Clean up resources"""
        self.vector_store.close()


# LangChain implementation for production use
def create_langchain_rag():
    """Create a LangChain RAG pipeline for disease diagnosis"""
    # Initialize components
    vector_store = DiseaseVectorStore()
    symptom_extractor = SymptomExtractor()
    
    # Create Groq LLM
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name="mixtral-8x7b-32768",
    )
    
    # Define retriever function
    def retrieve_similar_diseases(query):
        # Extract symptoms
        extracted_symptoms = symptom_extractor.extract_symptoms(query)
        symptom_text = ", ".join(extracted_symptoms) if extracted_symptoms else query
        
        # Retrieve similar diseases
        results = vector_store.similarity_search(symptom_text, top_k=5)
        
        # Format as documents
        docs = []
        for result in results:
            content = f"Disease: {result['disease']}\nSymptoms: {result['symptoms_text']}\nSimilarity: {result['similarity']:.4f}"
            docs.append(Document(page_content=content, metadata={"disease": result["disease"]}))
        
        return {
            "docs": docs,
            "extracted_symptoms": extracted_symptoms
        }
    
    # Define prompt template
    template = """
    You are a medical diagnostic assistant. Your task is to analyze the patient's symptoms and suggest possible diagnoses.
    
    PATIENT DESCRIPTION:
    {query}
    
    EXTRACTED SYMPTOMS:
    {extracted_symptoms}
    
    SIMILAR CASES FROM MEDICAL DATABASE:
    {docs}
    
    Based on the patient's description and the similar cases, provide the following:
    1. Most likely diagnoses (list up to 3 possibilities in order of likelihood)
    2. Explanation for each diagnosis
    3. Important symptoms that support each diagnosis
    4. Any critical warning signs the patient should be aware of
    5. General recommendations (e.g., see a doctor, home care, etc.)
    
    IMPORTANT: Always include a disclaimer that this is not a replacement for professional medical advice.
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Create RAG chain
    def format_docs(docs_dict):
        docs_text = "\n\n".join([doc.page_content for doc in docs_dict["docs"]])
        symptoms_text = ", ".join(docs_dict["extracted_symptoms"])
        return {"docs": docs_text, "extracted_symptoms": symptoms_text}
    
    rag_chain = (
        {"query": RunnablePassthrough()}
        | {"query": RunnablePassthrough(), **retrieve_similar_diseases}
        | format_docs
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain


if __name__ == "__main__":
    # Simple test
    rag = DiagnosisRAG()
    result = rag.diagnose("I've been having headaches and feeling dizzy for the past few days")
    print(json.dumps(result, indent=2))
    rag.close()