import fire, sys, json
from rich import print
from .pipeline import Pipeline

def main() -> None:
    try:
        fire.Fire(Pipeline, name="RAG")
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as e:
        print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("Program exit successfully")

if __name__ == "__main__":
    main()