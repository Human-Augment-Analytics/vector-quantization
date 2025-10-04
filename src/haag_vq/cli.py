import typer
from .benchmarks.run_benchmarks import run
from .benchmarks.sweep import sweep
from .visualization.plot import plot

app = typer.Typer()
app.command(name="run")(run)
app.command(name="sweep")(sweep)
app.command(name="plot")(plot)

def main():
    app()

if __name__ == "__main__":
    main()