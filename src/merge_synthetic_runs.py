import pathlib
import pandas as pd

container = pathlib.Path("src/results")


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
    except:
        # print(f"Could not load {file}")
        return None


def import_dataframe():
    dfs = []
    for folder in container.iterdir():
        if not folder.name.startswith("synthetic_") or not folder.is_dir():
            continue
        _, seed, depth, difficulty = folder.name.split("_")

        for method in folder.iterdir():
            if method.is_dir():
                for file in method.iterdir():
                    if file.name != "main.csv":
                        df = process_file(
                            file, int(seed), int(depth), difficulty, method.name
                        )
                        if df is not None:
                            dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def main():
    main_df = import_dataframe()
    main_df.to_parquet(container / "synthetic_summary.parquet")


if __name__ == "__main__":
    main()
