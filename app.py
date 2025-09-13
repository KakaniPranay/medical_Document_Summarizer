from flask import Flask, render_template, request, redirect, url_for, flash
from summarizer import HybridSummarizer
import os

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "supersecretkey123")

# instantiate summarizer (loads models lazily)
summ = HybridSummarizer()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        text = request.form.get("text", "").strip()
        if not text:
            flash("Please paste or upload medical text to summarize.", "warning")
            return redirect(url_for("index"))

        # method: 'hybrid' (extractive+abstractive), 'extractive' or 'abstractive'
        method = request.form.get("method", "hybrid")
        try:
            summary = summ.summarize(text, method=method)
        except Exception as e:
            # make failure graceful and return extractive fallback
            app.logger.exception("Error during summarization")
            flash(f"Error during summarization: {e}. Returning extractive summary.", "danger")
            summary = summ.summarize(text, method="extractive")

        return render_template("index.html", original=text, summary=summary, method=method)

    return render_template("index.html", original="", summary=None, method="hybrid")

if __name__ == "__main__":
    # allow port override
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, port=port)
