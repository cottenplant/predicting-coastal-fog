import model_search
import pandas as pd

import warnings
warnings.filterwarnings(action='ignore')

def run_search(n, data):
    df = pd.DataFrame()
    for i in range(0, n + 1):
        print("\nTEST # {}".format(i))
        results = model_search.main(i, data)
        df = pd.concat([df, results])
    summary = df.groupby('Algorithm').mean()
    print("\n\n===SUMMARY===")
    print(summary)
    summary.to_csv('compare_classifiers_summary.csv')


if __name__ == "__main__":
    run_search(1000, dataset)
