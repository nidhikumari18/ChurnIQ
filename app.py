from flask import Flask, jsonify, render_template

app = Flask(__name__)


@app.get("/")
def home():
    """Render the public landing page."""
    return render_template("index.html")


@app.get("/health")
def health():
    """Lightweight check that confirms the Flask process is responding."""
    return jsonify(status="ok", application="ChurnIQ")


if __name__ == "__main__":
    # Keep debug mode off by default. Enable it locally with FLASK_DEBUG=1.
    app.run(debug=app.config.get("DEBUG", False))
