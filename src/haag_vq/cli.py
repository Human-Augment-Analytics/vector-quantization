import typer
from .benchmarks.run_benchmarks import run

app = typer.Typer()
app.command(name="run")(run)

def main():
    app()

if __name__ == "__main__":
    main()