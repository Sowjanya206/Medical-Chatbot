# Medical-FAQ-chatbot

# Building a Medical FAQ chatbot using finetuned SBERT model. Dataset used was MEDQUAD

We have built this as our NLP course project. The Medsquad dataset contains 47k Question-Context-Answer Mapping. we utilised an Sentence transformer BERT model and finutuned it on our data to fetch the Most Accurate answer from the database given a Query. It works by creating sentence embedding for the Query question and checks the similarity for All the Questions in the database and get the answer of the most similar question. later we extended our project by developing ROBERTA and MEDBERT finetuned sentence transformer models. Elasticsearch was used for indexing and faster retrieval.

For Questions that aren't present in the database (similarity<30%) we used finetuned BERT to pass the context obtained from webscraping and getting the answer. this QA pair is later added to the database after user feedback.
