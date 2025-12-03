from pathlib import Path

from flask import Flask

from .api import bp_api
from .ui import bp_ui


def create_app() -> Flask:
    """
    Фабрика приложения: создаёт Flask, вешает блюпринты ui + api.
    Используем общие templates/static из корня проекта.
    """
    # path до корня проекта: fraud_app/.. → проект
    base_dir = Path(__file__).resolve().parent.parent
    templates_dir = base_dir / "templates"
    static_dir = base_dir / "static"

    app = Flask(
        __name__,
        template_folder=str(templates_dir),
        static_folder=str(static_dir),
    )

    # регаем блюпринты
    app.register_blueprint(bp_ui)          # фронт на "/"
    app.register_blueprint(bp_api, url_prefix="/api")  # api на "/api/..."

    return app