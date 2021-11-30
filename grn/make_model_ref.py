import argparse
import pandas as pd
from collections import defaultdict


def main(args):
    def_dict = defaultdict(list)
    prefix = "size_type_"
    for f in args.files:
        df = pd.read_csv(f).set_index("time")
        for col in df.columns:
            if not col.startswith(prefix):
                continue
            else:
                new_col = col[len(prefix):]
            def_dict[new_col].append(df[col])

    # mean
    mean_series = list()
    for k, v in def_dict.items():
        df = pd.concat(v).fillna(0)
        df_group = df.groupby(level=0).mean()

        mean_series.append(df_group)

    final_df = pd.concat(mean_series, axis=1).reset_index()
    final_df.to_csv(args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--files", nargs="*")
    parser.add_argument("-o", "--output")

    args = parser.parse_args()
    main(args)
