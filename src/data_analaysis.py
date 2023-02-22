import pandas as pd


def main():
    df = pd.read_csv("src/results/synthetic_summary.csv")
    print(df.shape)


if __name__ == "__main__":
    main()
