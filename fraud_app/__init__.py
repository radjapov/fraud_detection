def create_app():
    # Imported lazily: fraud_app.app pulls in the API blueprint, which loads model
    # artifacts (and prints) at import time. Helper modules such as the SHAP worker
    # import fraud_app.features / fraud_app.explain and must not pay for that.
    from .app import create_app as _create_app

    return _create_app()


__all__ = ["create_app"]
