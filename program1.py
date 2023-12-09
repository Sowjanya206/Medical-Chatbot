from flask import Flask, render_template, request
from fuzzywuzzy import fuzz
from googlesearch import search
import requests
from bs4 import BeautifulSoup
from transformers import BertForQuestionAnswering, BertTokenizer
from tqdm import tqdm
import pandas as pd
import torch


# Load BERT model and tokenizer
model = BertForQuestionAnswering.from_pretrained('deepset/bert-base-cased-squad2')
tokenizer = BertTokenizer.from_pretrained('deepset/bert-base-cased-squad2')

csv_file_path = 'MedQuad.csv'
medical_df = pd.read_csv(csv_file_path)
print("csv read")


def perform_web_scraping(user_question):
    # Perform a Google search
    search_results = list(search(user_question, num=10, stop=10, pause=2))

    # Process search results
    best_answer = None
    best_confidence = 0.0

    for i, result in enumerate(search_results):
        # Send a GET request to the URL
        try:
            response = requests.get(result)
            response.raise_for_status()
        except requests.exceptions.HTTPError as errh:
            continue
        except requests.exceptions.ConnectionError as errc:
            continue
        except requests.exceptions.Timeout as errt:
            continue
        except requests.exceptions.RequestException as err:
            continue

        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')  # Extract text from paragraphs

        # Concatenate paragraphs into a single text
        web_text = ' '.join([paragraph.get_text() for paragraph in paragraphs])

        # Tokenize and get model output
        inputs = tokenizer(web_text, return_tensors="pt", max_length=512, truncation=True)

        # Use the BERT model to get the answer
        outputs = model(**inputs)
        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits) + 1
        bert_answer = tokenizer.decode(inputs["input_ids"][0][answer_start:answer_end])

        # Calculate confidence based on similarity to the user's question
        confidence = fuzz.token_set_ratio(user_question.lower(), bert_answer.lower()) / 100.0

        # Update the best answer if confidence is higher
        if confidence > best_confidence:
            best_answer = bert_answer

    return best_answer
