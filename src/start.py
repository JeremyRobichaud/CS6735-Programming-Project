import glob
import logging
import sys
from datetime import datetime
import random

import pandas as pd

from algorithms.id3 import compute as id3_compute
from algorithms.adaboost import compute as adaboost_compute
from algorithms.naivesbayes import compute as nb_compute
from algorithms.ann import compute as ann_compute
from algorithms.council import compute as council_compute

from preprocessing import PreProcessor


def cross_fold(data_file_path, func, times=10, k_fold=5, name="TestingAlg", target_num=0):
    data_file_paths = glob.glob(data_file_path)
    for data_p in data_file_paths:
        logging.debug(f"Testing {name} on {data_p}...")
        pp = PreProcessor()
        df = pd.read_csv(data_p, header=None)
        target_attr = 0
        cols = list(df)
        indexable_cols = []
        drops = []

        if "car.data" in data_p or "breast-cancer-wisconsin.data" in data_p or "ecoli.data" in data_p:
            target_attr = len(cols) - 1

        if "breast-cancer-wisconsin.data" in data_p:
            indexable_cols = []
            drops = [0]

        if "car.data" in data_p:
            indexable_cols = []
            drops = []

        if "ecoli.data" in data_p:
            indexable_cols = [1, 2, 5, 6, 7]
            drops = [0]

        if "letter-recognition.data" in data_p:
            indexable_cols = []
            drops = []

        if "mushroom.data" in data_p:
            indexable_cols = []
            drops = []

        all_attribute_v = {}
        for c in cols:
            attrs = list(df[c].unique())
            if "?" in attrs:
                attrs.remove("?")
            all_attribute_v[c] = attrs
            assert "?" not in all_attribute_v[c]

        cols.pop(target_attr)
        for d in drops:
            cols.pop(d)
        # 10 Times
        total_acc = 0
        j = 0
        error = 0
        while j < times:
            k_fold_acc = 0
            # Shuffle the df
            cur_df = df.copy().sample(frac=1)
            assert len(all_attribute_v[target_attr]) == len(cur_df[target_attr].unique())
            all_attribute_values = all_attribute_v.copy()

            if 0 < target_num < len(all_attribute_values[target_attr]):
                df_length = 0
                flag = False
                counter = 0
                while int(df_length / k_fold) == 0:
                    targets = random.sample(list(all_attribute_values[target_attr]), target_num)
                    cur_df = cur_df[cur_df[target_attr].isin(targets)]
                    all_attribute_values[target_attr] = targets
                    df_length = len(cur_df)
                    counter += 1
                    if counter > 50:
                        flag = True
                        break
                if flag:
                    break

            times_tested = 0

            for i in range(k_fold):
                # 5-Fold Cross Validation
                df_length = len(cur_df)
                chunk_size = int(df_length / k_fold)

                acc = 0

                testing_df = cur_df.copy().iloc[i * chunk_size:(i + 1) * chunk_size, :]

                testing_df_without_answers = testing_df.copy().drop(columns=[target_attr])
                training_df = pd.concat([cur_df.copy(), testing_df.copy()]).drop_duplicates(keep=False)

                if len(list(testing_df[target_attr].unique())) < 2:
                    break

                # logging.debug(f"\t...[Fold {i + 1}] training PreProcessor")
                pp.train(training_df, target_attr, indexable_cols, drops)
                for c in indexable_cols:
                    all_attribute_values[c] = [i for i in range(len(cur_df[target_attr].unique()))]
                training_df = pp.process(training_df)

                # logging.debug(f"\t...[Fold {i + 1}] training {name}")
                try:
                    classifier = func(training_df.copy(), target_attr, cols.copy(), all_attribute_values)
                except Exception as err:
                    error += 1
                    logging.error(f"...[Fold {i + 1}] An Error Has Occurred, retrying...")
                    if error > 5:
                        logging.error(f"...[Fold {i + 1}] Too many errors has occurred, exiting...")
                        raise err
                    break
                testing_df_without_answers = pp.process(testing_df_without_answers)

                for index, row in testing_df_without_answers.iterrows():
                    if classifier.classify(row) == testing_df.loc[index][target_attr]:
                        acc += 1

                k_fold_acc += acc / len(testing_df)
                logging.debug(f"\t...[Fold {i + 1}] accuracy found {acc / len(testing_df)}")
                times_tested += 1

            if times_tested < k_fold:
                logging.debug(f"...[Iteration #{j + 1}] error occurred, redoing this iteration")
                continue

            error = 0
            total_acc += k_fold_acc / k_fold
            logging.debug(f"...[Iteration #{j + 1}] accuracy found {k_fold_acc / k_fold}")
            j += 1

        total_acc = total_acc / 10
        logging.info(f"{name} Accuracy of {data_p} = {total_acc}")


def start(k_fold):
    # cross_fold('./data/*.data', id3_compute, name="ID3", k_fold=k_fold,)
    # cross_fold('./data/*.data', adaboost_compute, name="Adaboost (2-way)", k_fold=k_fold, target_num=2)
    # cross_fold('./data/*.data', adaboost_compute, name="Adaboost (All)", k_fold=k_fold)
    # cross_fold('./data/*.data', nb_compute, name="Naive Bayes", k_fold=k_fold)
    cross_fold('./data/*.data', ann_compute, name="Neural Network", k_fold=k_fold)
    cross_fold('./data/*.data', council_compute, name="Council", k_fold=k_fold)

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
