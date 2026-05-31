from iris_client import run_query


def main():
    """Smoke test: connect to IRIS and print the result of a simple query."""
    result = run_query("SELECT $ZVERSION AS version")
    print("Connected to IRIS:")
    print(result)


if __name__ == "__main__":
    main()
