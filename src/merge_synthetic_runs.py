import pathlib
import pandas as pd
import pyarrow.parquet as pq

container = pathlib.Path("results")


def process_file(
    file: pathlib.Path, seed: int, depth: int, difficulty: str, method: str
):
    try:
        df = pd.read_csv(file)
        return df
    except Exception as e:
        print(e)
        print(f"Could not load {file}")
        return None


def import_dataframe(number_of_parts : int, part : int):
    n_dirs_in_container = sum([ 1 for f in container.iterdir() if f.is_dir() ])
    part_size = round(n_dirs_in_container / number_of_parts)
    start = part_size * part
    end = part_size * (part + 1) if part + 1 != number_of_parts else n_dirs_in_container + 1

    dfs_evolution = []
    for idx, folder in enumerate(container.iterdir()):
        if idx % 100 == 0:
            print(idx)
        if not folder.name.startswith("synthetic_") or not folder.is_dir() or idx < start or idx >= end:
            continue
        _, seed, depth, difficulty = folder.name.split("_")

        for method in folder.iterdir():
            if method.is_dir():
                for file in method.iterdir():
                    df_evolution = process_file(
                        file, int(seed), int(depth), difficulty, method.name
                    )
                    if df_evolution is not None:
                        dfs_evolution.append(df_evolution)
    return pd.concat(dfs_evolution, ignore_index=True)


def main():
    number_of_parts = 2
    file_locations = list()

    for part in range(number_of_parts):
        dfs_evolution = import_dataframe(number_of_parts, part)
        file_location = f"synthetic_evolution{part}.parquet"
        dfs_evolution.to_parquet(file_location)
        file_locations.append(file_location)

    print("Merging parquet files:", file_locations)
    with pq.ParquetWriter("synthetic_evolution.parquet", schema=pq.ParquetFile(file_locations[0]).schema_arrow) as writer:
        for file in file_locations:
            writer.write_table(pq.read_table(file))


if __name__ == "__main__":
    main()
