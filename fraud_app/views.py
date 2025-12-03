# fraud_app/views.py
from flask import Blueprint, jsonify, render_template

bp = Blueprint(
    "ui",
    __name__,
    url_prefix="",  # корневой префикс
)


@bp.get("/")
def index():
    # Можно пока просто заглушку или твой index.html
    return render_template("index.html")


@bp.get("/api/health")
def health():
    # Пример простого API-эндпоинта
    return jsonify({"status": "ok", "source": "blueprint"})
