# Databricks notebook source
# MAGIC %md
# MAGIC # Entity Search and Resolution using Databricks Vector Search
# MAGIC
# MAGIC This notebook demonstrates how to:
# MAGIC 1. Set up a vector search index on company names
# MAGIC 2. Perform similarity search to find potential entity matches
# MAGIC 3. Use an LLM to resolve entities and determine if candidates represent the same entity
# MAGIC
# MAGIC ## Prerequisites
# MAGIC - Databricks workspace with Vector Search enabled
# MAGIC - Company names dataset loaded in Unity Catalog

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Install Required Packages

# COMMAND ----------

# MAGIC %pip install -U --quiet databricks-sdk==0.36.0 databricks-agents mlflow-skinny==2.18.0 mlflow==2.18.0 mlflow[gateway]==2.18.0 databricks-vectorsearch langchain==0.2.1 langchain_core==0.2.5 langchain_community==0.2.4
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Configuration Parameters

# COMMAND ----------

# Configuration parameters - Modify these as needed
CATALOG = "henryk_dbdemos"
DATABASE = "db_finance" 
TABLE_NAME = "company_name"
VECTOR_SEARCH_ENDPOINT_NAME = "dbdemos_vs_endpoint"
EMBEDDING_MODEL_ENDPOINT = "databricks-gte-large-en"
LLM_MODEL_NAME = "databricks-meta-llama-3-3-70b-instruct"

# Derived parameters
TABLE_NAME_IDX = f"{TABLE_NAME}_idx"
SOURCE_TABLE_FULLNAME = f"{CATALOG}.{DATABASE}.{TABLE_NAME}"
VS_INDEX_FULLNAME = f"{CATALOG}.{DATABASE}.{TABLE_NAME_IDX}"

print(f"Source Table: {SOURCE_TABLE_FULLNAME}")
print(f"Vector Index: {VS_INDEX_FULLNAME}")
print(f"Endpoint: {VECTOR_SEARCH_ENDPOINT_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Explore the Dataset

# COMMAND ----------

# MAGIC %sql 
# MAGIC -- Examine the company names dataset
# MAGIC SELECT * FROM henryk_dbdemos.db_finance.company_name LIMIT 10

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Initialize Vector Search Client and Create Endpoint

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

# Initialize Vector Search client
vsc = VectorSearchClient(disable_notice=True)

# Create vector search endpoint (if it doesn't exist)
try:
    vsc.create_endpoint(name=VECTOR_SEARCH_ENDPOINT_NAME, endpoint_type="STANDARD")
    print(f"✅ Created endpoint: {VECTOR_SEARCH_ENDPOINT_NAME}")
except Exception as e:
    print(f"ℹ️ Endpoint might already exist: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Enable Change Data Feed on Source Table

# COMMAND ----------

# Enable Change Data Feed for real-time sync
spark.sql(f"ALTER TABLE {SOURCE_TABLE_FULLNAME} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
print(f"✅ Enabled Change Data Feed on {SOURCE_TABLE_FULLNAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Create Vector Search Index

# COMMAND ----------

from databricks.sdk import WorkspaceClient
import databricks.sdk.service.catalog as c

try:
    print(f"Creating index {VS_INDEX_FULLNAME} on endpoint {VECTOR_SEARCH_ENDPOINT_NAME}...")
    
    vsc.create_delta_sync_index(
        endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
        index_name=VS_INDEX_FULLNAME,
        source_table_name=SOURCE_TABLE_FULLNAME,
        pipeline_type="TRIGGERED",
        primary_key="id",
        embedding_source_column="name",  # The column containing company names
        embedding_model_endpoint_name=EMBEDDING_MODEL_ENDPOINT
    )
    
    print(f"✅ Successfully created vector index: {VS_INDEX_FULLNAME}")
    
except Exception as e:
    print(f"ℹ️ Index might already exist or there was an issue: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Test Similarity Search

# COMMAND ----------

def perform_similarity_search(query_text, num_results=5):
    """
    Perform similarity search using the vector index
    
    Args:
        query_text (str): The entity to search for
        num_results (int): Number of similar results to return
    
    Returns:
        list: List of similar company names with scores
    """
    try:
        results = vsc.get_index(VECTOR_SEARCH_ENDPOINT_NAME, VS_INDEX_FULLNAME).similarity_search(
            query_text=query_text,
            columns=["name"],
            num_results=num_results
        )
        
        data_array = results.get('result', {}).get('data_array', [])
        return data_array
        
    except Exception as e:
        print(f"❌ Error performing similarity search: {e}")
        return []

# Test the similarity search
test_query = "Pepsi"
print(f"Testing similarity search for: '{test_query}'")

search_results = perform_similarity_search(test_query, num_results=5)
print(f"Found {len(search_results)} similar entities:")
for i, result in enumerate(search_results, 1):
    print(f"  {i}. {result[0]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Entity Resolution with LLM

# COMMAND ----------

import json
from databricks.sdk import WorkspaceClient

def resolve_entity_with_llm(master_entity, candidates, model_name=LLM_MODEL_NAME):
    """
    Use LLM to determine if any candidate entities match the master entity
    
    Args:
        master_entity (str): The primary entity to match against
        candidates (list): List of candidate entity names
        model_name (str): Name of the LLM model to use
    
    Returns:
        dict: Entity resolution result with best match, score, and justification
    """
    
    # Initialize Databricks SDK client
    w = WorkspaceClient()
    openai_client = w.serving_endpoints.get_open_ai_client()
    
    # System prompt for entity resolution
    system_prompt = (
        "You are an expert entity resolution assistant.\n\n"
        "You will be provided with a master entity and a list of candidate entities. "
        "Your task is to determine if any of the candidates represent the SAME ENTITY "
        "as the master entity (e.g., exact name match, alias, abbreviation, or spelling variation).\n\n"
        "IMPORTANT RULES:\n"
        "- Do NOT choose candidates that are merely similar, competitors, or in the same industry\n"
        "- Only match if the candidate is truly the same entity (same company, person, etc.)\n"
        "- If none of the candidates are the same entity, respond with 'no_matching'\n"
        "- Always provide a confidence score (1–5) and a clear justification\n"
        "- Respond ONLY in the specified JSON schema\n"
    )
    
    # User prompt with entities
    user_prompt = f"""
Master Entity: {master_entity} \n\n
Candidate Entities: {candidates} \n\n

Please analyze if any candidate represents the same entity as the master entity.
"""
    
    # JSON response schema
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "entity_match_assessment",
            "description": (
                "Identify if any candidate entity represents the same entity as the master entity. "
                "If none match, respond with no_matching."
            ),
            "schema": {
                "type": "object",
                "properties": {
                    "best_match": {
                        "type": "string",
                        "description": "The candidate entity name that matches the master entity, or 'no_matching'."
                    },
                    "score": {
                        "type": "integer",
                        "description": "Confidence score from 1 (poor match) to 5 (exact match).",
                        "minimum": 1,
                        "maximum": 5
                    },
                    "justification": {
                        "type": "string",
                        "description": "Clear explanation of why this candidate matches or why no match was found."
                    }
                },
                "required": ["best_match", "score", "justification"],
                "strict": True
            }
        }
    }
    
    try:
        # Call LLM for entity resolution
        completion = openai_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=256,
            response_format=response_format
        )
        
        # Parse and return result
        content = completion.choices[0].message.content
        result = json.loads(content)
        return result
        
    except Exception as e:
        print(f"❌ Error in entity resolution: {e}")
        return {
            "best_match": "error",
            "score": 0,
            "justification": f"Error occurred: {str(e)}"
        }

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Complete Entity Resolution Pipeline

# COMMAND ----------

def complete_entity_resolution(master_entity, num_candidates=5, model_name=LLM_MODEL_NAME):
    """
    Complete pipeline: similarity search + LLM entity resolution
    
    Args:
        master_entity (str): The entity to resolve
        num_candidates (int): Number of similar candidates to retrieve
        model_name (str): LLM model for entity resolution
    
    Returns:
        dict: Complete resolution result
    """
    
    print(f"🔍 Starting entity resolution for: '{master_entity}'")
    print("-" * 60)
    
    # Step 1: Similarity search
    print("Step 1: Finding similar entities via vector search...")
    search_results = perform_similarity_search(master_entity, num_candidates)
    
    if not search_results:
        return {
            "master_entity": master_entity,
            "candidates": [],
            "resolution_result": {
                "best_match": "no_candidates",
                "score": 0,
                "justification": "No candidates found via similarity search"
            }
        }
    
    candidates = [result[0] for result in search_results]
    print(f"Found {len(candidates)} candidates: {candidates}")
    
    # Step 2: LLM entity resolution
    print(f"\nStep 2: Resolving entities using {model_name}...")
    resolution_result = resolve_entity_with_llm(master_entity, candidates, model_name)
    
    # Display results
    print(f"\n📊 RESOLUTION RESULTS:")
    print(f"Master Entity: {master_entity}")
    print(f"Best Match: {resolution_result['best_match']}")
    print(f"Confidence Score: {resolution_result['score']}/5")
    print(f"Justification: {resolution_result['justification']}")
    
    return {
        "master_entity": master_entity,
        "candidates": candidates,
        "resolution_result": resolution_result
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Example Usage and Testing

# COMMAND ----------

# Example 1: Test with "Pepsi"
result_pepsi = complete_entity_resolution("Pepsi", num_candidates=5, model_name=LLM_MODEL_NAME)

# COMMAND ----------

# Example 2: Test with another entity
result_coca_cola = complete_entity_resolution("Coca Cola", num_candidates=5, model_name=LLM_MODEL_NAME)

# COMMAND ----------

# Example 3: Test with a potentially non-matching entity
result_microsoft = complete_entity_resolution("Microsoft", num_candidates=5, model_name=LLM_MODEL_NAME)