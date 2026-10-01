import fire
from .pipeline import Pipeline

def main() -> None:
    fire.Fire(Pipeline)

if __name__ == "__main__":
    main()