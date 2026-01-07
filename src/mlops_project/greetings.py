import typer

app = typer.Typer()


@app.command()
def hello(count: int = 1):
    """Print hello message count times."""
    for _ in range(count):
        typer.echo("Hello, World!")


if __name__ == "__main__":
    app()
