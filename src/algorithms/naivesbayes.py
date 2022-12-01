# Naïve Bayes
import warnings
warnings.filterwarnings("ignore")


class NBClassifier:
    class _Attribute:
        def __init__(self, name, values):
            self.name = name
            self.values = values

    def __init__(self, training_data, target_name, target_values, k_smoothing=1):
        self._target_attr = self._Attribute(target_name, target_values)
        self._training_data = training_data
        self._training_attributes = {}
        for c in list(training_data):
            attribute_values = list(training_data[c].unique())
            self._training_attributes[c] = attribute_values
        self._training_data_len = len(training_data)
        self._k = k_smoothing
        self._probs = {}
        self._probs_given = {}
        self.train()

    def train(self):
        prob = {}
        for key in self._training_attributes.keys():
            prob[key] = {}
            for value in self._training_attributes[key]:
                # C(d)
                value_len = len(self._training_data.loc[self._training_data[key] == value])
                # P_Laplace(d, k) = (C(d)+k)/(N + k*|V|)
                prob[key][value] = (value_len + self._k) / \
                                   (self._training_data_len + (len(self._training_attributes[key]) * self._k))

            prob[key]["__UNK"] = (0 + self._k) / \
                                    (self._training_data_len + (len(self._training_attributes[key]) * self._k))
        self._probs = prob

    def prob_d(self, d, d_attr):
        assert self._probs
        assert d_attr in self._probs.keys()
        if d not in self._probs[d_attr]:
            return self._probs[d_attr]["__UNK"]

        # P_Laplace(d, k) = (C(d)+k)/(N + k*|V|)
        return self._probs[d_attr][d]

    def prob_d_given_h(self, d, d_attr, h, h_attr):
        # Check if stored previously
        if d_attr in self._probs_given.keys() \
                and d in self._probs_given[d_attr].keys() \
                and h_attr in self._probs_given[d_attr][d].keys() \
                and h in self._probs_given[d_attr][d][h_attr].keys():
            return self._probs_given[d_attr][d][h_attr][h]
        # P(D|h) = P(h|D)*P(h)/P(D)
        assert d_attr in self._training_attributes.keys() and h_attr in self._training_attributes.keys()

        # C(h ^ D)
        d_loc = self._training_data.loc[self._training_data[d_attr] == d]
        hd_len = len(d_loc[self._training_data[h_attr] == h])
        # C(D)
        d_len = len(d_loc)
        # P(h|D) = [C(h ^ D)+k]/[C(D) + kV]
        h_given_d = (hd_len + self._k) / (d_len + (self._k * len(self._training_attributes[d_attr])))
        # P(h)
        prob_h = self.prob_d(h, h_attr)
        # P(d)
        prob_d = self.prob_d(d, d_attr)

        # Store for future
        if d_attr not in self._probs_given.keys():
            self._probs_given[d_attr] = {}
        if d not in self._probs_given[d_attr].keys():
            self._probs_given[d_attr][d] = {}
        if h_attr not in self._probs_given[d_attr][d].keys():
            self._probs_given[d_attr][d][h_attr] = {}
        if h not in self._probs_given[d_attr][d][h_attr].keys():
            self._probs_given[d_attr][d][h_attr][h] = h_given_d * prob_h / prob_d

        return self._probs_given[d_attr][d][h_attr][h]

    def classify(self, row):
        # Naive Bayes: argmx(P(v)multi(P(a|v)))
        retval = ""
        best_prob = -1
        for target_value in self._target_attr.values:
            prob = self.prob_d(target_value, self._target_attr.name)
            for attr in self._training_attributes.keys():
                if attr not in list(row.index):
                    continue
                prob *= self.prob_d_given_h(row[attr], attr, target_value, self._target_attr.name)
            if prob >= best_prob:
                best_prob = prob
                retval = target_value

        return retval


def compute(df, target_attribute, _, all_attribute_values):
    classifier = NBClassifier(df, target_attribute, all_attribute_values[target_attribute], k_smoothing=0.00001)
    return classifier
