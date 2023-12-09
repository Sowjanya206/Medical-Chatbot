


import pandas as pd
df=pd.read_csv('merged_qa_dataset.csv')


df = df.dropna(subset=['Answer'])

# Reset the index of the DataFrame after removing rows
df = df.reset_index(drop=True)


df.drop('Category', axis =1, inplace=True)


# Commented out IPython magic to ensure Python compatibility.
# %pip install sentence-transformers

# Commented out IPython magic to ensure Python compatibility.
# %pip install spacy

# Commented out IPython magic to ensure Python compatibility.
# %pip install -U sentence-transformers

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("SBERT_finetuned")

# Commented out IPython magic to ensure Python compatibility.
# %pip install elasticsearch


from elasticsearch import Elasticsearch

es_client = Elasticsearch(
    ['http://localhost:9200'],
    http_auth=('Akshara', 'Akshara@123')
)

INDEX_NAME = "faq_bot_index"

EMBEDDING_DIMS = 768

def create_index() -> None:

    es_client.indices.delete(index=INDEX_NAME, ignore=404)

    es_client.indices.create(

        index=INDEX_NAME,

        ignore=400,

        body={

            "mappings": {

                "properties": {

                    "embedding": {

                        "type": "dense_vector",

                        "dims": EMBEDDING_DIMS,

                    },

                    "question": {

                        "type": "text",

                    },

                    "answer": {

                        "type": "text",

                    }

                }

            }

        }

    )

create_index()

from typing import List, Dict
def index_qa_pairs(qa_pairs: List[Dict[str, str]]) -> None:

    for qa_pair in qa_pairs:
      print(".")

      question = qa_pair["question"]

      answer = qa_pair["answer"]

      embedding = model.encode(question).tolist()


      data = {

            'embedding': embedding,

            'question': question,

            'answer': answer,

        }

      es_client.index(

            index=INDEX_NAME,

            body=data

        )


qa_pairs = []

for index, row in df.iterrows():
    qa_pair = {"question": str(row["Question"]), "answer": str(row["Answer"])}
    qa_pairs.append(qa_pair)


index_qa_pairs(qa_pairs)

ENCODER_BOOST = 20

def query_question(question: str, top_n: int=10) -> List[dict]:
    embedding = model.encode(question).tolist()
    es_result = es_client.search(
        index=INDEX_NAME,
        body={
            "from": 0,
            "size": top_n,
            "_source": ["question", "answer"],
            "query": {
                "script_score": {
                    "query": {
                        "match": {
                            "question": question
                        }
                    },
                    "script": {
                        "source": """
                            (cosineSimilarity(params.query_vector, "embedding") + 1)
                            * params.encoder_boost + _score
                        """,
                        "params": {
                            "query_vector": embedding,
                            "encoder_boost": ENCODER_BOOST,
                        },
                    },
                }
            }
        }
    )
    hits = es_result["hits"]["hits"]
    clean_result = []
    for hit in hits:
        clean_result.append({
            "question": hit["_source"]["question"],
            "answer": hit["_source"]["answer"],
            "score": hit["_score"],
        })
    return clean_result

