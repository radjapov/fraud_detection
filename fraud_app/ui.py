# fraud_app/ui.py
from flask import Blueprint, render_template

# Blueprint только для UI (HTML + статика)
bp_ui = Blueprint("ui", __name__)


@bp_ui.route("/")
def index():
    # Flask возьмет шаблон из template_folder, который указан в create_app()
    return render_template("index.html")