from flask import Flask, render_template, request, jsonify
from program1 import perform_web_scraping

from program2 import query_question

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    return get_Chat_response(input)


def get_Chat_response(text):
    answer=query_question(text)[0]['answer']
    score=query_question(text)[0]['score']
    if score> 0.35:
        return answer
    else:
        web_answer = perform_web_scraping(text)
        if web_answer is not None :
            return web_answer
        else:
            return "No suitable answer found."



if __name__ == '__main__':
    app.run(debug=True)







