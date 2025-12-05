from flask import Flask, request, render_template
from rag import query_with_summarization

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        query = request.form.get("query")
        answer = query_with_summarization(query, return_html=True)
        return render_template("result.html", query=query, answer=answer)

    return render_template("index.html")
    

if __name__ == "__main__":
    app.run(debug=True)
