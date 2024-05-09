#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('pip', 'install -q --upgrade --user google-cloud-aiplatform google-cloud-discoveryengine google-cloud-storage google-cloud-bigquery[pandas] google-cloud-bigquery-storage pandas ipywidgets')


# In[2]:


get_ipython().run_line_magic('load_ext', 'google.cloud.bigquery')


# In[1]:


# Automatically restart kernel after installs so that your environment can access the new packages
import IPython

app = IPython.Application.instance()
app.kernel.do_shutdown(True)


# In[1]:


from typing import List
import requests
import subprocess
import time

from google.api_core.client_options import ClientOptions
from google.api_core.exceptions import GoogleAPICallError
from google.cloud import bigquery
from google.cloud import discoveryengine_v1alpha as discoveryengine
from google.cloud import storage

from tqdm import tqdm  # to show a progress bar

import vertexai
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput

tqdm.pandas()


# In[2]:


# Define project information for Vertex AI
PROJECT_ID = "qwiklabs-gcp-02-5f752a71d416"  # @param {type:"string"}
LOCATION = "us-central1"  # @param {type:"string"}

# Initialize Vertex AI SDK
vertexai.init(project=PROJECT_ID, location=LOCATION)


# In[3]:


bq_client = bigquery.Client(project=PROJECT_ID)
query = f"""
SELECT
  DISTINCT 
  q.id,
  q.title,
  q.body,
  q.answer_count,
  q.comment_count,
  q.creation_date,
  q.last_activity_date,
  q.score,
  q.tags,
  q.view_count
FROM
  `bigquery-public-data.stackoverflow.posts_questions` AS q
WHERE
  q.score > 0
ORDER BY
  q.view_count DESC
LIMIT
  500;
"""

# Load the BQ Table into a Pandas Dataframe
df = bq_client.query(query).result().to_dataframe()

# Convert ID to String
df["id"] = df["id"].apply(str)

# examine the data
df.head()


# In[4]:


# Load the text embeddings model
model = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")


# In[5]:


# Get embeddings for a list of texts
def get_embeddings_wrapper(texts, batch_size: int = 5) -> List:
    embs = []
    for i in tqdm(range(0, len(texts), batch_size)):
        # Create embeddings optimized for document retrieval
        # (supported in textembedding-gecko@002 and later)
        result = model.get_embeddings(
            [
                TextEmbeddingInput(text=text, task_type="RETRIEVAL_DOCUMENT")
                for text in texts[i : i + batch_size]
            ]
        )
        embs.extend([e.values for e in result])
    return embs


# In[6]:


df["title_body"] = df["title"] + "\n" + df["body"]

df = df.assign(embedding=get_embeddings_wrapper(df.title_body))
df.head()


# In[7]:


BUCKET_NAME = "qwiklabs-gcp-02-5f752a71d416"
BUCKET_URI = "gs://qwiklabs-gcp-02-5f752a71d416"
REGION = "us-central1"
PROJECT_ID = "qwiklabs-gcp-02-5f752a71d416"


# In[8]:


get_ipython().system(' gsutil mb -l $REGION -p $PROJECT_ID $BUCKET_URI')


# In[9]:


get_ipython().run_cell_magic('bash', '', '\n# Set your Google Cloud Storage bucket name\nBUCKET_NAME="gs://qwiklabs-gcp-02-5f752a71d416"\n\n# Array of top-level directory names you want to create\nTOP_LEVEL_DIRECTORIES=("embeddings-stackoverflow")\n\n# Loop through the top-level array and create directories\nfor TOP_LEVEL_DIRECTORY in "${TOP_LEVEL_DIRECTORIES[@]}"; do\n  gsutil -m cp -r /dev/null "$BUCKET_NAME/$TOP_LEVEL_DIRECTORY/"\n\n  # Array of subdirectory names you want to create inside the top-level directory\n  SUBDIRECTORIES=("html")\n\n  # Loop through the subdirectories array and create subdirectories inside the top-level directory\n  for SUBDIRECTORY in "${SUBDIRECTORIES[@]}"; do\n    gsutil -m cp -r /dev/null "$BUCKET_NAME/$TOP_LEVEL_DIRECTORY/$SUBDIRECTORY/"\n  done\ndone\n\necho "Directories created successfully."\n')


# In[10]:


JSONL_MIME_TYPE = "application/jsonl"
HTML_MIME_TYPE = "text/html"

BUCKET_NAME = "qwiklabs-gcp-02-5f752a71d416"
DIRECTORY = "embeddings-stackoverflow"
BLOB_PREFIX = f"{DIRECTORY}/html/"

GCS_URI_PREFIX = f"gs://{BUCKET_NAME}/{BLOB_PREFIX}"

storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)


def scrape_question(question_url: str) -> str:
    response = requests.get(question_url)

    if response.status_code != 200 or not response.content:
        print(f"URL: {question_url} Code: {response.status_code}")
        return None

    print(f"Scraping {question_url}")

    link_title = response.url.split("/")[-1] + ".html"
    gcs_uri = f"{GCS_URI_PREFIX}{link_title}"

    # Upload HTML to Google Cloud Storage
    blob = bucket.blob(f"{BLOB_PREFIX}{link_title}")
    blob.upload_from_string(response.content, content_type=HTML_MIME_TYPE)
    time.sleep(1)
    return gcs_uri


# In[11]:


# Get the published URL from the ID
QUESTION_BASE_URL = "https://stackoverflow.com/questions/"
df["question_url"] = df["id"].apply(lambda x: f"{QUESTION_BASE_URL}{x}")

# Scrape HTML from stackoverflow.com and upload to GCS
df["gcs_uri"] = df["question_url"].apply(scrape_question)


# In[12]:


EMBEDDINGS_FIELD_NAME = "embedding_vector"


def format_row(row):
    return {
        "id": row["id"],
        "content": {"mimeType": HTML_MIME_TYPE, "uri": row["gcs_uri"]},
        "structData": {
            EMBEDDINGS_FIELD_NAME: row["embedding"],
            "title": row["title"],
            "body": row["body"],
            "question_url": row["question_url"],
            "answer_count": row["answer_count"],
            "creation_date": row["creation_date"],
            "score": row["score"],
        },
    }


vais_embeddings = (
    df.apply(format_row, axis=1)
    .to_json(orient="records", lines=True, force_ascii=False)
    .replace("\/", "/")  # To prevent escaping the / characters
)


# In[13]:


jsonl_filename = f"{DIRECTORY}/vais_embeddings.jsonl"
embeddings_file = f"gs://{BUCKET_NAME}/{jsonl_filename}"

blob = bucket.blob(jsonl_filename)
blob.upload_from_string(vais_embeddings, content_type=JSONL_MIME_TYPE)


# In[14]:


DATA_STORE_LOCATION = "global"

client_options = (
    ClientOptions(api_endpoint=f"{DATA_STORE_LOCATION}-discoveryengine.googleapis.com")
    if DATA_STORE_LOCATION != "global"
    else None
)


# In[15]:


def create_data_store(
    project_id: str, location: str, data_store_name: str, data_store_id: str
):
    # Create a client
    client = discoveryengine.DataStoreServiceClient(client_options=client_options)

    # Initialize request argument(s)
    data_store = discoveryengine.DataStore(
        display_name=data_store_name,
        industry_vertical="GENERIC",
        content_config="CONTENT_REQUIRED",
        solution_types=["SOLUTION_TYPE_SEARCH"],
    )

    request = discoveryengine.CreateDataStoreRequest(
        parent=discoveryengine.DataStoreServiceClient.collection_path(
            project_id, location, "default_collection"
        ),
        data_store=data_store,
        data_store_id=data_store_id,
    )
    operation = client.create_data_store(request=request)

    try:
        operation.result()
    except GoogleAPICallError:
        pass


def update_schema(
    project_id: str,
    location: str,
    data_store_id: str,
):
    client = discoveryengine.SchemaServiceClient(client_options=client_options)

    schema = discoveryengine.Schema(
        name=client.schema_path(project_id, location, data_store_id, "default_schema"),
        struct_schema={
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {
                EMBEDDINGS_FIELD_NAME: {
                    "type": "array",
                    "keyPropertyMapping": "embedding_vector",
                    "dimension": 768,
                    "items": {"type": "number"},
                }
            },
        },
    )

    operation = client.update_schema(
        request=discoveryengine.UpdateSchemaRequest(schema=schema)
    )

    print("Waiting for operation to complete...")

    response = operation.result()

    # Handle the response
    print(response)


def import_documents(
    project_id: str,
    location: str,
    data_store_id: str,
    gcs_uri: str,
):
    client = discoveryengine.DocumentServiceClient(client_options=client_options)

    # The full resource name of the search engine branch.
    # e.g. projects/{project}/locations/{location}/dataStores/{data_store_id}/branches/{branch}
    parent = client.branch_path(
        project=project_id,
        location=location,
        data_store=data_store_id,
        branch="default_branch",
    )

    request = discoveryengine.ImportDocumentsRequest(
        parent=parent,
        gcs_source=discoveryengine.GcsSource(input_uris=[gcs_uri]),
        # Options: `FULL`, `INCREMENTAL`
        reconciliation_mode=discoveryengine.ImportDocumentsRequest.ReconciliationMode.FULL,
    )

    # Make the request
    operation = client.import_documents(request=request)


def create_engine(
    project_id: str, location: str, data_store_name: str, data_store_id: str
):
    client = discoveryengine.EngineServiceClient(client_options=client_options)

    # Initialize request argument(s)
    config = discoveryengine.Engine.SearchEngineConfig(
        search_tier="SEARCH_TIER_ENTERPRISE", search_add_ons=["SEARCH_ADD_ON_LLM"]
    )

    engine = discoveryengine.Engine(
        display_name=data_store_name,
        solution_type="SOLUTION_TYPE_SEARCH",
        industry_vertical="GENERIC",
        data_store_ids=[data_store_id],
        search_engine_config=config,
    )

    request = discoveryengine.CreateEngineRequest(
        parent=discoveryengine.DataStoreServiceClient.collection_path(
            project_id, location, "default_collection"
        ),
        engine=engine,
        engine_id=engine.display_name,
    )

    # Make the request
    operation = client.create_engine(request=request)
    response = operation.result(timeout=90)


# In[16]:


DATA_STORE_NAME = "stackoverflow-embeddings"
DATA_STORE_ID = f"{DATA_STORE_NAME}-id"


# In[17]:


# Create a Data Store
create_data_store(PROJECT_ID, DATA_STORE_LOCATION, DATA_STORE_NAME, DATA_STORE_ID)

# Update the Data Store Schema for embeddings
update_schema(PROJECT_ID, DATA_STORE_LOCATION, DATA_STORE_ID)

# Import the embeddings JSONL file
import_documents(PROJECT_ID, DATA_STORE_LOCATION, DATA_STORE_ID, embeddings_file)

# Create a Search App and attach the Data Store
create_engine(PROJECT_ID, DATA_STORE_LOCATION, DATA_STORE_NAME, DATA_STORE_ID)


# In[18]:


access_token = (
    subprocess.check_output(["gcloud", "auth", "print-access-token"])
    .decode("utf-8")
    .strip()
)

response = requests.patch(
    url=f"https://discoveryengine.googleapis.com/v1alpha/projects/{PROJECT_ID}/locations/{DATA_STORE_LOCATION}/collections/default_collection/dataStores/{DATA_STORE_ID}/servingConfigs/default_search?updateMask=embeddingConfig,rankingExpression",
    headers={
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json; charset=utf-8",
        "X-Goog-User-Project": PROJECT_ID,
    },
    json={
        "name": f"projects/{PROJECT_ID}/locations/{DATA_STORE_LOCATION}/collections/default_collection/dataStores/{DATA_STORE_ID}/servingConfigs/default_search",
        "embeddingConfig": {"fieldPath": EMBEDDINGS_FIELD_NAME},
        "ranking_expression": "0.5 * relevance_score",
    },
)

print(response.text)


# In[19]:


def search_data_store(
    project_id: str,
    location: str,
    data_store_id: str,
    search_query: str,
) -> List[discoveryengine.SearchResponse]:
    # Create a client
    client = discoveryengine.SearchServiceClient(client_options=client_options)

    # The full resource name of the search engine serving config
    # e.g. projects/{project_id}/locations/{location}/dataStores/{data_store_id}/servingConfigs/{serving_config_id}
    serving_config = client.serving_config_path(
        project=project_id,
        location=location,
        data_store=data_store_id,
        serving_config="default_config",
    )

    # Optional: Configuration options for search
    # Refer to the `ContentSearchSpec` reference for all supported fields:
    # https://cloud.google.com/python/docs/reference/discoveryengine/latest/google.cloud.discoveryengine_v1.types.SearchRequest.ContentSearchSpec
    content_search_spec = discoveryengine.SearchRequest.ContentSearchSpec(
        # For information about snippets, refer to:
        # https://cloud.google.com/generative-ai-app-builder/docs/snippets
        snippet_spec=discoveryengine.SearchRequest.ContentSearchSpec.SnippetSpec(
            return_snippet=True
        ),
        # For information about search summaries, refer to:
        # https://cloud.google.com/generative-ai-app-builder/docs/get-search-summaries
        summary_spec=discoveryengine.SearchRequest.ContentSearchSpec.SummarySpec(
            summary_result_count=5,
            include_citations=True,
            ignore_adversarial_query=True,
            ignore_non_summary_seeking_query=True,
        ),
    )

    # Refer to the `SearchRequest` reference for all supported fields:
    # https://cloud.google.com/python/docs/reference/discoveryengine/latest/google.cloud.discoveryengine_v1.types.SearchRequest
    request = discoveryengine.SearchRequest(
        serving_config=serving_config,
        query=search_query,
        page_size=10,
        content_search_spec=content_search_spec,
        query_expansion_spec=discoveryengine.SearchRequest.QueryExpansionSpec(
            condition=discoveryengine.SearchRequest.QueryExpansionSpec.Condition.AUTO,
        ),
        spell_correction_spec=discoveryengine.SearchRequest.SpellCorrectionSpec(
            mode=discoveryengine.SearchRequest.SpellCorrectionSpec.Mode.AUTO
        ),
    )

    response = client.search(request)
    return response


# In[22]:


search_query = "How do I create an array in Java?"

response = search_data_store(
    PROJECT_ID, DATA_STORE_LOCATION, DATA_STORE_ID, search_query
)

print(f"Summary: {response.summary.summary_text}")


# In[ ]:




