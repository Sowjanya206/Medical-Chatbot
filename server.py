from twilio.twiml.messaging_response import MessagingResponse
from flask import Flask, request

# Function to get the answer using your question-answering code
from twilio.rest import Client
from transformers import BertForQuestionAnswering, BertTokenizer
import torch
from fuzzywuzzy import fuzz
from googlesearch import search
import requests
from bs4 import BeautifulSoup

# Twilio credentials
account_sid = 'AC9b7d1bb160e02bcc4b59354c36a71d8f'
auth_token = 'bbd534a1eabb1cd2b22f593491a5fb6b'
twilio_phone_number = '14155238886'

# Create a Twilio client
client = Client(account_sid, auth_token)

# Load BERT model and tokenizer
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

# Function to fetch answers from the web
def get_answer_from_web(user_question):
    # Perform a Google search
    search_results = list(search(user_question, num=10, stop=10, pause=2))

    # Process search results
    best_answer = None
    best_confidence = 0.0

    for i, result in enumerate(search_results):
        print(f"\nFetching information from result {i + 1}: {result}")

        # Send a GET request to the URL
        try:
            response = requests.get(result)
            response.raise_for_status()
        except requests.exceptions.HTTPError as errh:
            print(f"HTTP Error: {errh}")
            continue
        except requests.exceptions.ConnectionError as errc:
            print(f"Error Connecting: {errc}")
            continue
        except requests.exceptions.Timeout as errt:
            print(f"Timeout Error: {errt}")
            continue
        except requests.exceptions.RequestException as err:
            print(f"Error: {err}")
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

        # Print the answer from web scraping
        print(f"Answer from web scraping: {bert_answer}")

        # Calculate confidence based on similarity to the user's question
        confidence = fuzz.token_set_ratio(user_question.lower(), bert_answer.lower()) / 100.0

        # Update the best answer if confidence is higher
        if confidence > best_confidence:
            best_answer = bert_answer
            best_confidence = confidence

    return best_answer, best_confidence

# Flask app for receiving messages
app = Flask(__name__)

# Define a route for receiving WhatsApp messages
@app.route('/webhook', methods=['POST'])
def webhook():
    incoming_msg = request.values.get('Body', '').lower()
    resp = MessagingResponse()

    # Get the answer using your question-answering code
    answer, _ = get_answer_from_web(incoming_msg)

    # Respond with the answer
    resp.message(answer)

    return str(resp)

# Run the Flask app
if __name__ == '__main__':
    app.run(port=5000)
