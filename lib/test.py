import model_search
import pandas as pd


def run_search(n):
    df = pd.DataFrame()
    for i in range(0, n + 1):
        results = model_search.main(i)
        df = pd.concat([df, results])
    summary = df.groupby('Algorithm').mean()
    print("\n\n===SUMMARY===")
    print(summary)


if __name__ == "__main__":
    run_search(1000)
