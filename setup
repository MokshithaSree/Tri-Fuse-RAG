from huggingface_hub import login
print("Please paste your Hugging Face Access Token below.")
print("Get it here: https://huggingface.co/settings/tokens")
login()
import os

print("--- DATABASE SETUP ---")
print("NOTE: In Colab, 'localhost' will NOT work for Postgres/Neo4j.")
print("Use a cloud Postgres (Neon, Supabase, etc.) and Neo4j Aura, or leave blank to disable.")

# Press Enter to use defaults where mentioned
os.environ['PG_HOST'] = input('Postgres host (leave blank to disable SQL): ') or ""
os.environ['PG_DB']   = input('Postgres database name (default: postgres): ') or "postgres"
os.environ['PG_USER'] = input('Postgres user (leave blank to disable SQL): ') or ""
os.environ['PG_PASS'] = input('Postgres password: ') or ""
os.environ['PG_PORT'] = "5432"

print("\n--- GRAPH SETUP ---")
os.environ['NEO4J_URI']  = input('Neo4j URI (e.g., bolt://... or neo4j+s://... , blank to disable): ') or ""
os.environ['NEO4J_USER'] = input('Neo4j user (default: neo4j): ') or "neo4j"
os.environ['NEO4J_PASS'] = input('Neo4j password: ') or ""

print("\n✅ Credentials saved (empty host/URI means that DB type will be disabled).")
