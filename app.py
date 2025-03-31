from flask import Flask, render_template, request, redirect, session
from user_db import (
    register_user, authenticate, get_user_history, get_user_slides,
    add_history, add_slide, get_user
)
from Functions.pipelines import Explanation_Pipeline, Dictionary_Pipeline
from Functions.languages import get_language_list

app = Flask(__name__)
app.secret_key = "supersecretkey"


@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user_id = request.form["user_id"]
        password = request.form["password"]
        user = authenticate(user_id, password)
        if user:
            session["user_id"] = user_id
            session["name"] = user["name"]
            return redirect("/dashboard")
        return "Invalid credentials"
    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        user_id = request.form["user_id"]
        name = request.form["name"]
        password = request.form["password"]
        mother_tongue = request.form["mother_tongue"]
        register_user(user_id, name, password, mother_tongue)
        session["user_id"] = user_id
        session["name"] = name
        return redirect("/dashboard")
    languages = get_language_list()
    return render_template("register.html", languages=languages)


@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        return redirect("/")
    return render_template("dashboard.html", name=session["name"])


@app.route("/explain", methods=["GET", "POST"])
def explain():
    if "user_id" not in session:
        return redirect("/")

    user = get_user(session["user_id"])
    default_lang = user["mother_tongue"]
    languages = get_language_list()
    result = None

    if request.method == "POST":
        input_text = request.form["input_text"]
        language = request.form["language"]
        result = Explanation_Pipeline(input_text, language)
        add_history(session["user_id"], input_text, result["elaboration"], result["output"])

    history = get_user_history(session["user_id"])
    return render_template("explain.html", result=result, history=history, languages=languages, default_lang=default_lang)


@app.route("/dictionary", methods=["GET", "POST"])
def dictionary():
    if "user_id" not in session:
        return redirect("/")

    user = get_user(session["user_id"])
    default_lang = user["mother_tongue"]
    languages = get_language_list()
    result = None

    if request.method == "POST":
        word = request.form["word"]
        language = request.form["language"]
        result = Dictionary_Pipeline(word, language)

        add_slide(
            session["user_id"],
            result["input_word"],
            result["input_word_translation"],
            result["meaning"],
            result["meaning_translation"]
        )

    slides = get_user_slides(session["user_id"])
    return render_template("dictionary.html", result=result, slides=slides, languages=languages, default_lang=default_lang)


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")


if __name__ == "__main__":
    app.run(debug=True)
