import typer
from .benchmarks.run_benchmarks import run
from .benchmarks.sweep import sweep
from .benchmarks.streaming_sweep import streaming_sweep
from .benchmarks.precompute_ground_truth import precompute_ground_truth
from .visualization.plot import plot

app = typer.Typer()
app.command(name="run")(run)
app.command(name="sweep")(sweep)
app.command(name="streaming-sweep")(streaming_sweep)
app.command(name="precompute-gt")(precompute_ground_truth)
app.command(name="plot")(plot)

def main():
    app()

if __name__ == "__main__":
    main()