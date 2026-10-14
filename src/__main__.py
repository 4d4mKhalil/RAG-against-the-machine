import fire, sys
from rich import print
from .pipeline import Pipeline

def main() -> None:
    try:
        fire.Fire(Pipeline, name="RAG")
    except (ValueError, FileNotFoundError) as e:
        print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()