import glob
import logging
import sys
from datetime import datetime

import pandas as pd

from algorithms.id3 import compute as id3_compute
from algorithms.naivesbayes import compute as nb_compute


def cross_fold(data_file_path, func, times=10, k_fold=5, name="TestingAlg"):
    data_file_paths = glob.glob(data_file_path)
    for data_p in data_file_paths:
        logging.debug(f"Implementing for {data_p}...")
        df = pd.read_csv(data_p, header=None)
        target_attr = 0
        cols = list(df)

        if "car.data" in data_p or "breast-cancer-wisconsin.data" in data_p or "ecoli.data" in data_p:
            target_attr = len(cols) - 1

        all_attribute_values = {}
        for c in cols:
            all_attribute_values[c] = df[c].unique()

        cols.pop(target_attr)
        # 10 Times
        total_id3_acc = 0
        for j in range(times):
            # Shuffle the df
            df = df.sample(frac=1)

            # 5-Fold Cross Validation
            df_length = len(df)
            chunk_size = int(df_length / k_fold)
            k_fold_id3_acc = 0

            for i in range(k_fold):
                acc = 0

                testing_df = df.iloc[i * chunk_size:(i + 1) * chunk_size, :]
                testing_df_without_answers = testing_df.copy().drop(columns=[target_attr])
                training_df = df.drop(range(i * chunk_size, (i + 1) * chunk_size))

                classifier = func(training_df, target_attr, cols.copy(), all_attribute_values)

                for index, row in testing_df_without_answers.iterrows():
                    if classifier.classify(row) == testing_df.loc[index][target_attr]:
                        acc += 1

                k_fold_id3_acc += acc / len(testing_df)
                logging.debug(f"\t...[Fold {i + 1}] accuracy found {acc / len(testing_df)}")

            total_id3_acc += k_fold_id3_acc / k_fold
            logging.debug(f"...[Iteration #{j + 1}] accuracy found {k_fold_id3_acc / k_fold}")

        total_id3_acc = total_id3_acc / 10
        logging.info(f"{name} Accuracy of {data_p} = {total_id3_acc}")


def start(k_fold):

    # cross_fold('./data/*.data', id3_compute, name="ID3", k_fold=k_fold)
    cross_fold('./data/*.data', nb_compute, name="Naive Bayes", k_fold=k_fold)

    pass


if __name__ == "__main__":
    now = datetime.now()  # current date and time

    log_format = "%(asctime)s [%(levelname)s]: %(message)s"
    logging.basicConfig(format=log_format, level=logging.DEBUG,
                        datefmt="%H:%M:%S", handlers=[
            logging.FileHandler(f"./logs/{now.strftime('%d-%m-%Y-%H-%M-%S')}.log"),
            logging.StreamHandler(sys.stdout)
        ])
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("psaw").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    try:
        start(k_fold=5)
    except Exception as e:
        logging.error(e)
        raise e
