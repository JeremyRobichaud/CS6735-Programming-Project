import sys


class PreProcessor:
    def __init__(self):
        self.indexable_attr = []
        self.indexes = {}
        self.drop_columns = []

    def _t_indexables_attr(self, df, target_value, indexable_attr):
        self.indexable_attr = indexable_attr
        for ia in self.indexable_attr:
            self.indexes[ia] = []
        t_values = df[target_value].unique()

        retval = {}

        for ia in self.indexable_attr:
            avgs = []
            for value in t_values:
                avgs.append([0, value])
            count = 0
            # This will give [(80, A), (60, B), (70, C)]
            for value in t_values:
                for index, rows in df.loc[df[target_value] == value].iterrows():
                    count += 1
                    for v in avgs:
                        if v[1] == value:
                            v[0] += rows[ia]
                for v in avgs:
                    if v[1] == value:
                        v[0] /= count

            # This will give: [(60, B), (70, C), (80, A)]
            avgs.sort(key=lambda tup: tup[0])

            # {'index_attr': [(60, B), (70, C), (80, A)]}
            retval[ia] = avgs

        self.indexes = retval

    def train(self, df, target_value, indexable_attr=None, drops=None):
        if indexable_attr:
            self._t_indexables_attr(df, target_value, indexable_attr)
        if drops:
            self.drop_columns = drops

    # There's 3 things we need to do during preprocessing:
    # 1- Remove all Unknown Data "?"
    # 2- Transform all indexibles into yes/no based on some division of the data
    # 3- Remove non-important columns
    def process(self, df_real):

        df = df_real.copy()

        # 1- Remove all Unknown Data "?" by using most seen
        columns = list(df.columns)
        counter = {}
        question_marks = []
        for c in columns:
            counter[c] = {}
        for index, row in df.iterrows():
            for c in columns:
                if row[c] == "?":
                    question_marks.append((index, c))
                    continue
                if row[c] not in counter[c]:
                    counter[c][row[c]] = 1
                else:
                    counter[c][row[c]] += 1
        for c in columns:
            most_seen_count = -1
            most_seen = ""
            for value in counter[c].keys():
                if counter[c][value] > most_seen_count:
                    most_seen_count = counter[c][value]
                    most_seen = value
            counter[c] = most_seen
        for row_index, column in question_marks:
            df.at[row_index, column] = counter[column]

        # 2- Transform all indexibles attributes into 0/1/2... based on some division of the data
        for ia in self.indexable_attr:
            # [(60, B), (70, C), (80, A)]
            distribution = self.indexes[ia]
            for row_index, row in df.iterrows():
                # 72.8
                rvalue = row[ia]
                min_v = -1 * sys.maxsize
                retval = -1
                # ---0----|65|---1----|75|---2----
                for i in range(len(distribution)-1):
                    max_v = (distribution[i][0] + distribution[i+1][0]) / 2
                    if min_v < rvalue <= max_v:
                        retval = i
                    min_v = max_v

                if retval == -1:
                    retval = len(distribution)-1

                df.at[row_index, ia] = retval

        # 3- Remove non-important columns
        for dc in self.drop_columns:
            del df[dc]

        return df
