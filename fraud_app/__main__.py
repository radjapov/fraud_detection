import os

from .app import create_app


def main() -> None:
    app = create_app()
    # Localhost by default; Docker sets FRAUD_HOST=0.0.0.0 to publish the port.
    host = os.environ.get("FRAUD_HOST", "127.0.0.1")
    port = int(os.environ.get("FRAUD_PORT", "5001"))
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    main()
