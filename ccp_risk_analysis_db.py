# Databricks notebook source
# MAGIC %md # CCP Risk Management News Analysis PoC
# MAGIC
# MAGIC This notebook demonstrates an end-to-end batch inference solution for CCP risk management:
# MAGIC - Classifies financial news headlines for relevance to market impact
# MAGIC - Identifies which exposure assets are mentioned in relevant headlines
# MAGIC - Maps news events to potential clearing member exposures

# COMMAND ----------

# MAGIC %md ## Environment Setup

# COMMAND ----------

from databricks.sdk import WorkspaceClient
import time

w = WorkspaceClient()
user_email = w.current_user.me().display_name
username = user_email.split("@")[0]
default_schema_name = username.replace(" ", "_").lower()

# COMMAND ----------

# Configuration widgets
dbutils.widgets.dropdown("endpoint_name", "databricks-meta-llama-3-1-8b-instruct", 
                        ["databricks-meta-llama-3-1-8b-instruct","databricks-meta-llama-3-3-70b-instruct"])
dbutils.widgets.text("catalog_name", "users", "Data UC Catalog")
dbutils.widgets.text("schema_name", default_schema_name, "Data UC Schema")
dbutils.widgets.text("table_name", "batch_sentiment_data", "Data UC Table")

endpoint_name = dbutils.widgets.get("endpoint_name")
catalog_name = dbutils.widgets.get("catalog_name")
schema_name = dbutils.widgets.get("schema_name")
table_name = dbutils.widgets.get("table_name")

print(f"Using endpoint: {endpoint_name}")
print(f"Data source: {catalog_name}.{schema_name}.{table_name}")

# COMMAND ----------

# Preview the data
display(
    spark.sql(
        f"SELECT * FROM {catalog_name}.{schema_name}.{table_name} LIMIT 10"
    )
)

# COMMAND ----------

# MAGIC %md ## Step 1: News Relevance Classification
# MAGIC
# MAGIC First, we classify each headline as relevant (1) or not relevant (0) for market impact

# COMMAND ----------

# Schema for relevance classification
relevance_schema = """
{
    "type": "json_schema",
    "json_schema": {
        "name": "relevance_classification",
        "schema": {
            "type": "object",
            "properties": {
                "relevant": { 
                    "type": "string",
                    "enum": ["0", "1"]
                }
            }
        },
        "strict": true
    }
}
"""

print("Step 1: Classifying news relevance...")
start_time = time.time()

relevance_results = spark.sql(f"""
    SELECT 
        Headline,
        ai_query(
            '{endpoint_name}',
            CONCAT(
                'Classify this financial news headline as relevant (1) or not relevant (0) for market impact and risk management. ',
                'Consider headlines relevant if they discuss: market movements, economic indicators, central bank actions, ',
                'corporate earnings, geopolitical events, or financial sector developments. ',
                'Respond with just 1 or 0: ',
                Headline
            ),
            responseFormat => '{relevance_schema}'
        ) AS relevance_pred_raw,
        CAST(get_json_object(relevance_pred_raw, '$.relevant') AS INT) AS is_relevant
    FROM {catalog_name}.{schema_name}.{table_name}
""")

# Cache the results for reuse
relevance_results.cache()
display(relevance_results)

end_time = time.time()
print(f"Step 1 execution time: {end_time - start_time:.2f} seconds")

# COMMAND ----------

# Create temporary view first, then check relevance distribution
relevance_results.createOrReplaceTempView("relevance_temp")

relevance_stats = spark.sql("""
    SELECT 
        is_relevant,
        COUNT(*) as count,
        ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as percentage
    FROM relevance_temp
    GROUP BY is_relevant
    ORDER BY is_relevant
""")

display(relevance_stats)

# COMMAND ----------

# MAGIC %md ## Step 2: Exposure Asset Classification
# MAGIC
# MAGIC For relevant headlines, identify which CCP exposure assets are mentioned

# COMMAND ----------

# Schema for asset classification
asset_schema = """
{
    "type": "json_schema",
    "json_schema": {
        "name": "asset_classification",
        "schema": {
            "type": "object",
            "properties": {
                "exposure_asset": { 
                    "type": "string",
                    "enum": [
                        "Eurostoxx50", 
                        "VSTOXX", 
                        "Bund_Futures", 
                        "DAX", 
                        "CAC40", 
                        "EUR_USD", 
                        "Oil", 
                        "iTraxx_Europe",
                        "None"
                    ]
                }
            }
        },
        "strict": true
    }
}
"""

print("Step 2: Classifying exposure assets for relevant headlines...")
start_time = time.time()

# Only process relevant headlines (is_relevant = 1)
asset_results = spark.sql(f"""
    SELECT 
        Headline,
        is_relevant,
        CASE 
            WHEN is_relevant = 1 THEN
                ai_query(
                    '{endpoint_name}',
                    CONCAT(
                        'Analyze this financial headline and identify which market exposure asset it most closely relates to. ',
                        'Choose from: Eurostoxx50, VSTOXX, Bund_Futures, DAX, CAC40, EUR_USD, Oil, iTraxx_Europe. ',
                        'If the headline does not clearly relate to any of these specific assets, respond with "None". ',
                        'Consider: Eurostoxx50/DAX/CAC40 for European equity indices, VSTOXX for volatility, ',
                        'Bund_Futures for German bonds, EUR_USD for currency, Oil for commodities, ',
                        'iTraxx_Europe for credit derivatives. ',
                        'Headline: ',
                        Headline
                    ),
                    responseFormat => '{asset_schema}'
                )
            ELSE NULL
        END AS asset_pred_raw,
        CASE 
            WHEN is_relevant = 1 THEN 
                get_json_object(
                    ai_query(
                        '{endpoint_name}',
                        CONCAT(
                            'Analyze this financial headline and identify which market exposure asset it most closely relates to. ',
                            'Choose from: Eurostoxx50, VSTOXX, Bund_Futures, DAX, CAC40, EUR_USD, Oil, iTraxx_Europe. ',
                            'If the headline does not clearly relate to any of these specific assets, respond with "None". ',
                            'Consider: Eurostoxx50/DAX/CAC40 for European equity indices, VSTOXX for volatility, ',
                            'Bund_Futures for German bonds, EUR_USD for currency, Oil for commodities, ',
                            'iTraxx_Europe for credit derivatives. ',
                            'Headline: ',
                            Headline
                        ),
                        responseFormat => '{asset_schema}'
                    ), 
                    '$.exposure_asset'
                )
            ELSE 'N/A'
        END AS exposure_asset
    FROM relevance_temp
""")

display(asset_results)

end_time = time.time()
print(f"Step 2 execution time: {end_time - start_time:.2f} seconds")

# COMMAND ----------

# MAGIC %md ## Final Results: CCP Risk Management Analysis

# COMMAND ----------

# Create temporary view for asset results first
asset_results.createOrReplaceTempView("asset_results_temp")

# Create final summary table
final_results = spark.sql("""
    SELECT 
        Headline,
        is_relevant,
        exposure_asset,
        CASE 
            WHEN is_relevant = 1 AND exposure_asset != 'None' AND exposure_asset != 'N/A' 
            THEN 'HIGH_PRIORITY'
            WHEN is_relevant = 1 AND (exposure_asset = 'None' OR exposure_asset = 'N/A')
            THEN 'MEDIUM_PRIORITY'
            ELSE 'LOW_PRIORITY'
        END AS risk_priority,
        CURRENT_TIMESTAMP() as analysis_timestamp
    FROM asset_results_temp
    ORDER BY 
        CASE 
            WHEN is_relevant = 1 AND exposure_asset != 'None' AND exposure_asset != 'N/A' THEN 1
            WHEN is_relevant = 1 AND (exposure_asset = 'None' OR exposure_asset = 'N/A') THEN 2
            ELSE 3
        END,
        Headline
""")
display(final_results)

# COMMAND ----------

# Summary statistics
summary_stats = spark.sql("""
    SELECT 
        'Total Headlines' as metric,
        COUNT(*) as count
    FROM asset_results_temp
    
    UNION ALL
    
    SELECT 
        'Relevant Headlines' as metric,
        SUM(CASE WHEN is_relevant = 1 THEN 1 ELSE 0 END) as count
    FROM asset_results_temp
    
    UNION ALL
    
    SELECT 
        'Headlines with Asset Mapping' as metric,
        SUM(CASE WHEN is_relevant = 1 AND exposure_asset != 'None' AND exposure_asset != 'N/A' THEN 1 ELSE 0 END) as count
    FROM asset_results_temp
""")

print("=== CCP Risk Management Analysis Summary ===")
display(summary_stats)

# COMMAND ----------

# Asset distribution for relevant headlines
asset_distribution = spark.sql("""
    SELECT 
        exposure_asset,
        COUNT(*) as headline_count,
        ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as percentage
    FROM asset_results_temp
    WHERE is_relevant = 1
    GROUP BY exposure_asset
    ORDER BY headline_count DESC
""")

print("=== Exposure Asset Distribution (Relevant Headlines Only) ===")
display(asset_distribution)

# COMMAND ----------

# MAGIC %md ## Save Results for Further Analysis

# COMMAND ----------

# Save the final results to a new table for CCP risk analysts
output_table_name = f"{table_name}_risk_analysis"

final_results.write \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(f"{catalog_name}.{schema_name}.{output_table_name}")

print(f"Results saved to: {catalog_name}.{schema_name}.{output_table_name}")