import pathlib
import pandas as pd

container = pathlib.Path("src/results")

def process_main(
    file: pathlib.Path, seed: int, depth: int, difficulty: str, method: str
):
    try:
        df = pd.read_csv(file)
        df.columns = [
            "seed", 
            "configuration_name", 
            "grammar",
            "fitness difficulty",
            "representation",
            "maximum depth",
            "fitness",
            "Execution Time",
            "Peak memory usage",
            "Number of Generations", 
            "Fitness of the first generation",
            "HP Population size",
            "HP Elitism size",
            "HP Probability of Crossover", 
            "HP Probability of Mutation", 
            "HP Novelty size", 
            "GP tournament size",
            "G Minimum Depth", 
            "G Maximum Depth",
            "G Number of Non-terminals",
            "G Production occurrences", 
            "G Recursive Productions", 
            "G Non-terminal Count",
            "G Recursive Non-terminal Count",
            "G Average Productions per Non-terminal",
            "Non-terminals per Production" ]
        df["Original Seed"] = seed
        df["Requested Depth"] = depth
        df["Fitness Difficulty"] = difficulty
        df["Method"] = method
        return df
    except Exception as e:
        print(e)
        print(f"Could not load {file}")
        return None

def process_file(
    file: pathlib.Path, seed: int, depth: int, difficulty: str, method: str
):
    try:
        df = pd.read_csv(file)
        df["Seed"] = seed
        df["Requested Depth"] = depth
        df["Fitness Difficulty"] = difficulty
        df["Method"] = method
        return df
    except Exception as e:
        print(e)
        print(f"Could not load {file}")
        return None


def import_dataframe():
    dfs_metadata = []
    dfs_evolution = []
    for folder in container.iterdir():
        if not folder.name.startswith("synthetic_") or not folder.is_dir():
            continue
        _, seed, depth, difficulty = folder.name.split("_")

        for method in folder.iterdir():
            if method.is_dir():
                for file in method.iterdir():
                    if file.name == "main.csv":
                        df_metadata = process_main(file, int(seed), int(depth), difficulty, method.name)
                        if df_metadata is not None:
                            dfs_metadata.append(df_metadata)
                    else:
                        df_evolution = process_file(
                            file, int(seed), int(depth), difficulty, method.name
                        )
                        if df_evolution is not None:
                            dfs_evolution.append(df_evolution)
    return (pd.concat(dfs_evolution, ignore_index=True), pd.concat(dfs_metadata, ignore_index=True))


def main():
    dfs_evolution, dfs_metadata = import_dataframe()
    dfs_evolution.to_parquet(container / "synthetic_evolution.parquet")
    dfs_metadata.to_parquet(container / "synthetic_metadata.parquet")


if __name__ == "__main__":
    main()
