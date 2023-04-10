import pathlib
import pandas as pd

container = pathlib.Path("../results")


def process_file(file: pathlib.Path, seed: int, depth: int, difficulty: str, method: str):
    try:
        df = pd.read_csv(file)
        return df
    except Exception as e:
        print(e)
        print(f"Could not load {file}")
        return None


def import_dataframe():
    dfs_evolution = []
    for folder in container.iterdir():
        if not folder.name.startswith("synthetic_") or not folder.is_dir():
            continue
        _, seed, depth, difficulty = folder.name.split("_")

        for method in folder.iterdir():
            if method.is_dir():
                for file in method.iterdir():
                    df_evolution = process_file(file, int(seed), int(depth), difficulty, method.name)
                    if df_evolution is not None:
                        dfs_evolution.append(df_evolution)
    return pd.concat(dfs_evolution, ignore_index=True)


def main():
    dfs_evolution = import_dataframe()
    dfs_evolution.to_parquet(container / "synthetic_evolution.parquet")


if __name__ == "__main__":
    main()
