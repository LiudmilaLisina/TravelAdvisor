from flask import Flask, render_template, request
from agent import agent_executor
app = Flask(__name__)

chat_history = []

def get_response(user_input):
    result = agent_executor.invoke({
        "input": (user_input)
    })
    return result["output"]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = request.form['message']
        answer = get_response(user_input)
        chat_history.append(("user", user_input))
        chat_history.append(("bot", answer))
    return render_template('chat.html', chat_history=chat_history)

if __name__ == '__main__':
    app.run(debug=True)
