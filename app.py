from flask import Flask, render_template, request
from Functions.pipelines import Explanation_Pipeline
from user_db import register_user, add_history, get_history

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    history = []
    username = ""

    if request.method == "POST":
        username = request.form["username"]
        input_text = request.form["input_text"]
        language = request.form["language"]

        register_user(username)
        result = Explanation_Pipeline(input_text, language)

        entry = f"{input_text} → {result['output']}"
        add_history(username, entry)
        history = get_history(username)

    return render_template("index.html", result=result, history=history, username=username)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
