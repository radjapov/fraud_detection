from .app import create_app


def main() -> None:
    app = create_app()
    # порт можешь поменять, если нужно
    app.run(host="0.0.0.0", port=5001, debug=False)


if __name__ == "__main__":
    main()
