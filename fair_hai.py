import argparse
from human_ai_interactions_data import haiid
from sklearn.metrics import roc_auc_score, accuracy_score, r2_score
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import confusion_matrix
from sklearn.calibration import calibration_curve
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
import pickle

pd.options.mode.chained_assignment = None
np.set_printoptions(precision=3)
pd.options.display.float_format = "{:,.6f}".format
path = "plots/"


def categorize_label_art(label):
    if label in ["renaissance", "baroque"]:
        return "traditional"
    else:
        return "modern"


def categorize_label_city(label):
    if label in ["NewYork", "Chicago"]:
        return "eastern"
    else:
        return "western"


class Experiment:
    # class constructor, sets experiment's parameters for task
    def __init__(self, name, h_bins, b_bins, random_state):
        self.task_name = name
        self.h_bins = h_bins
        self.b_bins = b_bins
        self.h_bin_boundaries = []
        self.min_datapoints = 50
        self.task_data = None
        self.prob_y1_h_b = None
        self.df_cell_prob = None
        self.mass_y1_h_b = None
        self.prob_pi1_h_b = None
        self.mass_pi1_h_b = None

        self.df_metrics = None
        self.decision_model_female = None
        self.decision_model_male = None

        self.set_task_data()
        # generate decision model
        self.generate_decision_model(random_state)

        # compute probability matrix for cells
        self.compute_cell_prob_matrix()
        # run experiment
        self.run_experiment()

    def fair_aware_multicalibration(
        self,  # The above code is not valid Python code. It seems to
        # contain some random text ("alpha") and comments ("
        alpha,
        w1=1,
        w2=0,
    ):
        lambda_param = 1 / self.b_bins
        df = self.task_data

        # Get unique values from h_bin and b_bin columns
        # df_male = df[df["gender"] == "female"]
        # h_bin_values = np.sort(df["h_bin"].unique())
        # b_bin_values = np.sort(df["b_bin"].unique())

        # group S=1 conduct only alpha-multicalibration
        update = True
        all_b_bins = np.sort(df["b_bin"].unique())
        all_h_bins = np.sort(df["h_bin"].unique())
        while update:
            h_bin_values = np.sort(df["h_bin"].unique())
            b_bin_values = np.sort(df["b_bin"].unique())
            df_male = df[df["gender"] == "male"]
            update = False
            # Iterate over each combination of h_bin and b_bin
            for h_index, h_bin in enumerate(h_bin_values):
                df_male_h_bin = df_male[df_male["h_bin"] == h_bin]
                for b_index, b_bin in enumerate(b_bin_values):
                    # Filter the DataFrame for the current combination of h_bin and b_bin
                    df_male_h_bin_b_bin = df_male[
                        (df_male["h_bin"] == h_bin) & (df_male["b_bin"] == b_bin)
                    ]
                    if (
                        len(df_male_h_bin_b_bin)
                        <= alpha * lambda_param * len(df_male_h_bin)
                        or len(df_male_h_bin_b_bin) <= self.min_datapoints // 2
                    ):
                        continue
                    b_mean = df_male_h_bin_b_bin["b"].mean()
                    r_value = self.prob_y1_h_b[1].iat[b_index, h_index]
                    if abs(r_value - b_mean) > alpha:
                        update = True
                        df.loc[df_male_h_bin_b_bin.index, "b"] = df_male_h_bin_b_bin[
                            "b"
                        ].apply(lambda x: min(max(x + (r_value - b_mean), 0), 1))

            df["b_bin"] = (df["b"] // lambda_param) + 1
            df.loc[df["b_bin"] == self.b_bins + 1, ["b_bin"]] = self.b_bins
            df["b_bin"] *= lambda_param
            df["b_bin"] = df["b_bin"].round(3)

            self.prob_y1_h_b = [
                df[df["gender"] == "female"]
                .groupby(by=["b_bin", "h_bin"])["y"]
                .mean()
                .unstack()
                .reindex(index=all_b_bins, columns=all_h_bins, fill_value=0),
                df[df["gender"] == "male"]
                .groupby(by=["b_bin", "h_bin"])["y"]
                .mean()
                .unstack()
                .reindex(index=all_b_bins, columns=all_h_bins, fill_value=0),
            ]

        df_male = df[df["gender"] == "male"]
        for b_index, b_bin in enumerate(b_bin_values):
            df_male_b_bin = df_male[df_male["b_bin"] == b_bin]
            b_mean = df_male_b_bin["b"].mean()
            df.loc[df_male_b_bin.index, "b"] = b_mean

        df["b_bin"] = (df["b"] // lambda_param) + 1
        df.loc[df["b_bin"] == self.b_bins + 1, ["b_bin"]] = self.b_bins
        df["b_bin"] *= lambda_param
        df["b_bin"] = df["b_bin"].round(3)

        self.prob_y1_h_b = [
            df[df["gender"] == "female"]
            .groupby(by=["b_bin", "h_bin"])["y"]
            .mean()
            .unstack()
            .reindex(index=all_b_bins, columns=all_h_bins, fill_value=0),
            df[df["gender"] == "male"]
            .groupby(by=["b_bin", "h_bin"])["y"]
            .mean()
            .unstack()
            .reindex(index=all_b_bins, columns=all_h_bins, fill_value=0),
        ]
        # group S=0 conduct  fairness-aware alpha-multicalibration
        # df_male = df[df["gender"] == "male"]

        # Get unique values from h_bin and b_bin columns
        # df_female = df[df["gender"] == "female"]
        # h_bin_values = np.sort(df_female["h_bin"].unique())
        # b_bin_values = np.sort(df_female["b_bin"].unique())

        update = True
        while update:
            h_bin_values = np.sort(df["h_bin"].unique())
            b_bin_values = np.sort(df["b_bin"].unique())

            df_female = df[df["gender"] == "female"]
            update = False
            # Iterate over each combination of h_bin and b_bin
            for h_index, h_bin in enumerate(h_bin_values):
                df_female_h_bin = df_female[df_female["h_bin"] == h_bin]
                for b_index, b_bin in enumerate(b_bin_values):
                    # Filter the DataFrame for the current combination of h_bin and b_bin
                    df_female_h_bin_b_bin = df_female[
                        (df_female["h_bin"] == h_bin) & (df_female["b_bin"] == b_bin)
                    ]
                    if (
                        len(df_female_h_bin_b_bin)
                        <= alpha * lambda_param * len(df_female_h_bin)
                        or len(df_male_h_bin_b_bin) <= self.min_datapoints // 2
                    ):
                        continue
                    b_mean = df_female_h_bin_b_bin["b"].mean()
                    r_value = self.prob_y1_h_b[0].iat[b_index, h_index]
                    f_value = self.prob_y1_h_b[1].iat[b_index, h_index]
                    sum_value = w1 * r_value + w2 * f_value

                    if abs(sum_value - b_mean) > alpha:
                        update = True
                        df.loc[df_female_h_bin_b_bin.index, "b"] = (
                            df_female_h_bin_b_bin["b"].apply(
                                lambda x: min(max(x + (sum_value - b_mean), 0), 1)
                            )
                        )

            df["b_bin"] = (df["b"] // lambda_param) + 1
            df.loc[df["b_bin"] == self.b_bins + 1, ["b_bin"]] = self.b_bins
            df["b_bin"] *= lambda_param
            df["b_bin"] = df["b_bin"].round(3)

            self.prob_y1_h_b = [
                df[df["gender"] == "female"]
                .groupby(by=["b_bin", "h_bin"])["y"]
                .mean()
                .unstack()
                .reindex(index=all_b_bins, columns=all_h_bins, fill_value=0),
                df[df["gender"] == "male"]
                .groupby(by=["b_bin", "h_bin"])["y"]
                .mean()
                .unstack()
                .reindex(index=all_b_bins, columns=all_h_bins, fill_value=0),
            ]

        df_female = df[df["gender"] == "female"]
        for b_index, b_bin in enumerate(b_bin_values):
            df_female_b_bin = df_female[df_female["b_bin"] == b_bin]
            b_mean = df_female_b_bin["b"].mean()
            df.loc[df_female_b_bin.index, "b"] = b_mean

        df["b_bin"] = (df["b"] // lambda_param) + 1
        df.loc[df["b_bin"] == self.b_bins + 1, ["b_bin"]] = self.b_bins
        df["b_bin"] *= lambda_param
        df["b_bin"] = df["b_bin"].round(3)

        self.prob_y1_h_b = [
            df[df["gender"] == "female"]
            .groupby(by=["b_bin", "h_bin"])["y"]
            .mean()
            .unstack()
            .reindex(index=all_b_bins, columns=all_h_bins, fill_value=0),
            df[df["gender"] == "male"]
            .groupby(by=["b_bin", "h_bin"])["y"]
            .mean()
            .unstack()
            .reindex(index=all_b_bins, columns=all_h_bins, fill_value=0),
        ]

        # update decision model and h+AI column
        df.loc[df["gender"] == "female", "h+AI"] = self.decision_model_female.predict(
            df[df["gender"] == "female"][["h", "b"]]
        )
        df.loc[df["gender"] == "male", "h+AI"] = self.decision_model_male.predict(
            df[df["gender"] == "male"][["h", "b"]]
        )

        self.task_data = df

    def multicalibration(self, alpha):
        lambda_param = 1 / self.b_bins
        df = self.task_data

        update = True

        all_h_bin_values = np.sort(df["h_bin"].unique())
        all_b_bin_values = np.sort(df["b_bin"].unique())

        while update:
            h_bin_values = np.sort(df["h_bin"].unique())
            b_bin_values = np.sort(df["b_bin"].unique())
            update = False
            # Iterate over each combination of h_bin and b_bin
            for h_index, h_bin in enumerate(h_bin_values):
                df_h_bin = df[df["h_bin"] == h_bin]
                for b_index, b_bin in enumerate(b_bin_values):
                    # Filter the DataFrame for the current combination of h_bin and b_bin
                    df_h_bin_b_bin = df[(df["h_bin"] == h_bin) & (df["b_bin"] == b_bin)]
                    if (
                        len(df_h_bin_b_bin) <= alpha * lambda_param * len(df_h_bin)
                        or len(df_h_bin_b_bin) <= self.min_datapoints
                    ):
                        continue
                    b_mean = df_h_bin_b_bin["b"].mean()
                    r_value = self.df_cell_prob.iat[b_index, h_index]
                    if abs(r_value - b_mean) > alpha:
                        update = True
                        df.loc[df_h_bin_b_bin.index, "b"] = df_h_bin_b_bin["b"].apply(
                            lambda x: min(max(x + (r_value - b_mean), 0), 1)
                        )

            df["b_bin"] = (df["b"] // lambda_param) + 1
            df.loc[df["b_bin"] == self.b_bins + 1, ["b_bin"]] = self.b_bins
            df["b_bin"] *= lambda_param
            df["b_bin"] = df["b_bin"].round(3)

            self.df_cell_prob = (
                df.groupby(by=["b_bin", "h_bin"])["y"]
                .mean()
                .unstack()
                .reindex(index=all_b_bin_values, columns=all_h_bin_values, fill_value=0)
            )

        for b_index, b_bin in enumerate(b_bin_values):
            df_b_bin = df[df["b_bin"] == b_bin]
            b_mean = df_b_bin["b"].mean()
            df.loc[df_b_bin.index, "b"] = b_mean

        df["b_bin"] = (df["b"] // lambda_param) + 1
        df.loc[df["b_bin"] == self.b_bins + 1, ["b_bin"]] = self.b_bins
        df["b_bin"] *= lambda_param
        df["b_bin"] = df["b_bin"].round(3)

        self.df_cell_prob = (
            df.groupby(by=["b_bin", "h_bin"])["y"]
            .mean()
            .unstack()
            .reindex(index=all_b_bin_values, columns=all_h_bin_values, fill_value=0)
        )

        # update decision model and h+AI column
        df.loc[df["gender"] == "female", "h+AI"] = self.decision_model_female.predict(
            df[df["gender"] == "female"][["h", "b"]]
        )
        df.loc[df["gender"] == "male", "h+AI"] = self.decision_model_male.predict(
            df[df["gender"] == "male"][["h", "b"]]
        )

        self.task_data = df

    def generate_decision_model(self, random_state):
        df = self.task_data
        # female decision model
        # X = df[df["gender"] == "female"][["h", "b"]]
        # y = df[df["gender"] == "female"]["h+AI"]
        X = df[["h", "b"]]
        y = df["h+AI"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, random_state=random_state
        )

        model_female = MLPClassifier(
            hidden_layer_sizes=(20),
            learning_rate_init=0.1,
            max_iter=500,
            random_state=random_state,
        )

        model_female.fit(X_train, y_train)

        y_pred = model_female.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("decision model_female fitting accuracy:", accuracy)
        self.decision_model_female = model_female

        # male decision model
        # X = df[df["gender"] == "male"][["h", "b"]]
        # y = df[df["gender"] == "male"]["h+AI"]
        # X_train, X_test, y_train, y_test = train_test_split(
        #     X, y, test_size=0.1, random_state=random_state
        # )

        # model_male = MLPClassifier(
        #     hidden_layer_sizes=(20),
        #     learning_rate_init=0.1,
        #     max_iter=500,
        #     random_state=random_state,
        # )

        # model_male.fit(X_train, y_train)

        # y_pred = model_male.predict(X_test)
        # accuracy = accuracy_score(y_test, y_pred)
        # print("decision model_male fitting accuracy:", accuracy)
        self.decision_model_male = model_female

    # retrieve data in from of (h,b,y,t) for the defined task,
    # h =human risk estimate, b=model risk estimate, y= outcome, t=decision (response of human after seeing b)
    def set_task_data(self):
        # load all data
        df = haiid.load_dataset("./human_ai_interactions_data")

        # get specified task data
        task_df = haiid.load_task(df, self.task_name)
        if self.task_name == "census":
            self.task_data = self.preprocess_data(task_df, ">=50k")
        elif self.task_name == "sarcasm":
            self.task_data = self.preprocess_data(task_df, "sarcasm")
        else:
            self.task_data = self.preprocess_data(task_df)

        self.discretize_human_estimates()

        return self.task_data

    # map advice in [-1,1] to probabilities in [0,1] by assigning event Y=1 to label specified by label_1
    # if label_1==None, assigns event Y=1 to a random label per task instance
    def preprocess_data(self, data, label_1=None):

        # filter data for 'geographic region' and 'perceived accuracy'
        df = data.loc[
            (data["geographic_region"] == "United States")
            & (data["perceived_accuracy"] == 80)
        ]
        # df = data
        df = df[df["gender"] != "prefer not to say"]
        # select relevant data columns
        df = df[
            [
                "task_instance_id",
                "participant_id",
                "correct_label",
                "advice",
                "response_1",
                "response_2",
                "gender",
            ]
        ]

        # assign event Y=1
        if label_1 == None:
            # assigning event Y=1 to a random label per task instance
            # randomness needed since there are 4 labels accross tasks, but each task has two labels -> assign one of them to event Y=1
            np.random.seed(320)
            task_ids = df["task_instance_id"].unique()
            df_label = pd.DataFrame(task_ids, columns=["task_instance_id"])
            df_label["y"] = np.random.choice(2, len(task_ids)).astype(int)
            df = df.merge(df_label, how="left", on="task_instance_id")
        else:
            # assigning event Y=1 to label specified by label_1
            # The code is creating a new column "y" in the DataFrame `df` where the values are 1 if the
            # value in the column "correct_label" is equal to `label_1`, and 0 otherwise.
            df["y"] = (df["correct_label"] == label_1).astype(int)

        # compute mapping from [-1,1] to [0,1]
        df[["b", "h", "h+AI"]] = (
            df.loc[:, ["advice", "response_1", "response_2"]] + 1
        ) / 2.0
        df.loc[df["y"] == 0, ["b", "h", "h+AI"]] = (
            1 - df.loc[df["y"] == 0, ["b", "h", "h+AI"]]
        )

        # print data instances used
        # print(self.task_name)
        # print('all regions: ', data['geographic_region'].unique())
        # print('region used: ', 'United States')
        # print('number of participants: ', data['participant_id'].unique().size)
        # print('number of participants used: ', df['participant_id'].unique().size)
        # print('all datapoints: ',data.shape[0])
        # print('used datapoints: ',df.shape[0])

        # add
        # df["h+AI"] = 0.5 * df["h"] + 0.5 * df["b"]
        df["h+AI"] = np.where(df["h+AI"] < 0.5, 0, 1)
        return df[
            ["task_instance_id", "participant_id", "h", "b", "y", "h+AI", "gender"]
        ]
        # discretize human estimates into n_bins bins with aprroximately the same mass

    def discretize_human_estimates(self):

        # sort calibration data by human estimate h ascending
        df = self.task_data.copy()
        df = df.sort_values(by=["h"])

        # split calibration data into uniform sized bins
        # find bin boundaries
        if self.h_bins == 2:
            bin_bounds = [0.5, 1]
        else:
            df_split = np.array_split(df, self.h_bins)
            # find maximum value of h in each bin
            bin_bounds = []
            for df_h in df_split:
                max_h = round(df_h.loc[:, "h"].max(), 2)
                bin_bounds.append(max_h)

        # set value of h in calibration data to maximum value in each bin
        df.loc[
            (df["h"] < bin_bounds[0]) | np.isclose(df["h"], bin_bounds[0]), ["h_bin"]
        ] = bin_bounds[0]
        for i in range(1, len(bin_bounds)):
            df.loc[
                ((df["h"] < bin_bounds[i]) | np.isclose(df["h"], bin_bounds[i]))
                & (df["h"] > bin_bounds[i - 1]),
                ["h_bin"],
            ] = bin_bounds[i]

        # set data and h bin boundaries
        self.task_data = df
        self.h_bin_boundaries = bin_bounds

    # discretize human estimates into n_bins bins with aprroximately the same mass
    def discretize_human_estimates_group(self):
        # sort calibration data by human estimate h ascending
        df = self.task_data.copy()
        # Split the DataFrame into two groups based on the 'gender' attribute
        df_female = df[df["gender"] == "female"]
        df_male = df[df["gender"] == "male"]

        self.p_sensitive = [len(df_female) / len(df), len(df_male) / len(df)]

        df_female = df_female.sort_values(by=["h"])
        df_male = df_male.sort_values(by=["h"])

        # split calibration data into uniform sized bins
        # find bin boundaries
        if self.h_bins == 2:
            bin_bounds_female = [0.5, 1]
            bin_bounds_male = [0.5, 1]
        else:
            df_female_split = np.array_split(df_female, self.h_bins)
            df_male_split = np.array_split(df_male, self.h_bins)
            # find maximum value of h in each bin
            bin_bounds_female = []
            for df_h in df_female_split:
                max_h = round(df_h.loc[:, "h"].max(), 2)
                bin_bounds_female.append(max_h)

            bin_bounds_male = []
            for df_h in df_male_split:
                max_h = round(df_h.loc[:, "h"].max(), 2)
                bin_bounds_male.append(max_h)

        # set value of h in calibration data to maximum value in each bin
        df_female.loc[
            (df_female["h"] < bin_bounds_female[0])
            | np.isclose(df_female["h"], bin_bounds_female[0]),
            ["h_bin"],
        ] = bin_bounds_female[0]

        df_female.loc[
            (df_female["h"] < bin_bounds_female[0])
            | np.isclose(df_female["h"], bin_bounds_female[0]),
            ["h_bin_level"],
        ] = 0

        df_male.loc[
            (df_male["h"] < bin_bounds_male[0])
            | np.isclose(df_male["h"], bin_bounds_male[0]),
            ["h_bin"],
        ] = bin_bounds_male[0]

        df_male.loc[
            (df_male["h"] < bin_bounds_male[0])
            | np.isclose(df_male["h"], bin_bounds_male[0]),
            ["h_bin_level"],
        ] = 0

        for i in range(1, len(bin_bounds_female)):
            df_female.loc[
                (
                    (df_female["h"] < bin_bounds_female[i])
                    | np.isclose(df_female["h"], bin_bounds_female[i])
                )
                & (df_female["h"] > bin_bounds_female[i - 1]),
                ["h_bin"],
            ] = bin_bounds_female[i]

            df_female.loc[
                (
                    (df_female["h"] < bin_bounds_female[i])
                    | np.isclose(df_female["h"], bin_bounds_female[i])
                )
                & (df_female["h"] > bin_bounds_female[i - 1]),
                ["h_bin_level"],
            ] = i

        for i in range(1, len(bin_bounds_male)):
            df_male.loc[
                (
                    (df_male["h"] < bin_bounds_male[i])
                    | np.isclose(df_male["h"], bin_bounds_male[i])
                )
                & (df_male["h"] > bin_bounds_male[i - 1]),
                ["h_bin"],
            ] = bin_bounds_male[i]

            df_male.loc[
                (
                    (df_male["h"] < bin_bounds_male[i])
                    | np.isclose(df_male["h"], bin_bounds_male[i])
                )
                & (df_male["h"] > bin_bounds_male[i - 1]),
                ["h_bin_level"],
            ] = i

        # set data and h bin boundaries
        df = pd.concat([df_male, df_female]).reset_index(drop=True)
        self.task_data = df
        self.h_bin_boundaries = [bin_bounds_female, bin_bounds_male]

    def discretize_AI_estimates(self):
        df = self.task_data
        lambda_param = 1 / self.b_bins

        # partition model risk estimate space into B bins,
        # set new column in df with the max bin value (indicates in which bin each data point is))
        df["b_bin"] = (df["b"] // lambda_param) + 1
        df.loc[df["b_bin"] == self.b_bins + 1, ["b_bin"]] = self.b_bins
        df["b_bin"] *= lambda_param
        df["b_bin"] = df["b_bin"].round(3)

    # compute probabilities and density mass of each bin (h,b)
    def compute_cell_prob_matrix(self):
        df = self.task_data

        lambda_param = 1 / self.b_bins

        # partition model risk estimate space into B bins,
        # set new column in df with the max bin value (indicates in which bin each data point is))
        df["b_bin"] = (df["b"] // lambda_param) + 1
        df.loc[df["b_bin"] == self.b_bins + 1, ["b_bin"]] = self.b_bins
        df["b_bin"] *= lambda_param
        df["b_bin"] = df["b_bin"].round(3)

        # drop original model risk estimates
        # df = df.drop(columns=["b", "h+AI"])
        # compute probability Y=1 and density mass in each bin
        all_b_bins = np.sort(df["b_bin"].unique())
        all_h_bins = np.sort(df["h_bin"].unique())

        self.df_cell_prob = (
            df.groupby(by=["b_bin", "h_bin"])["y"]
            .mean()
            .unstack()
            .reindex(index=all_b_bins, columns=all_h_bins, fill_value=0)
            .fillna(0)
        )

        self.prob_y1_h_b = [
            df[df["gender"] == "female"]
            .groupby(by=["b_bin", "h_bin"])["y"]
            .mean()
            .unstack()
            .reindex(index=all_b_bins, columns=all_h_bins, fill_value=0)
            .fillna(0),
            df[df["gender"] == "male"]
            .groupby(by=["b_bin", "h_bin"])["y"]
            .mean()
            .unstack()
            .reindex(index=all_b_bins, columns=all_h_bins, fill_value=0)
            .fillna(0),
        ]

        self.mass_y1_h_b = [
            df[df["gender"] == "female"]
            .groupby(by=["b_bin", "h_bin"])["y"]
            .count()
            .unstack()
            .reindex(index=all_b_bins, columns=all_h_bins, fill_value=0)
            .fillna(0),
            df[df["gender"] == "male"]
            .groupby(by=["b_bin", "h_bin"])["y"]
            .count()
            .unstack()
            .reindex(index=all_b_bins, columns=all_h_bins, fill_value=0)
            .fillna(0),
        ]

        # self.proportion_h_b = [
        #     df[df["gender"] == "female"]
        #     .groupby(by=["b_bin", "h_bin"])
        #     .size()
        #     .unstack(),
        #     df[df["gender"] == "male"].groupby(by=["b_bin", "h_bin"]).size().unstack(),
        # ]

        # compute probability PI=1 and density mass in each bin
        # self.prob_pi1_h_b = [
        #     df[df["gender"] == "female"]
        #     .groupby(by=["b_bin", "h_bin"])["h+AI"]
        #     .mean()
        #     .unstack(),
        #     df[df["gender"] == "male"]
        #     .groupby(by=["b_bin", "h_bin"])["h+AI"]
        #     .mean()
        #     .unstack(),
        # ]
        # self.mass_pi1_h_b = [
        #     df[df["gender"] == "female"]
        #     .groupby(by=["b_bin", "h_bin"])["h+AI"]
        #     .count()
        #     .unstack(),
        #     df[df["gender"] == "male"]
        #     .groupby(by=["b_bin", "h_bin"])["h+AI"]
        #     .count()
        #     .unstack(),
        # ]

    # computes expected and average alignment error
    def check_alignment(self):
        # compute max and average alignment violations
        max_aligment = 0.0
        sum_aligment = 0.0
        max_aligment_s = 0.0
        sum_aligment_s = 0.0
        num_summants = 0.0
        disaligned_cells = set({})
        disaligned_cells_s = set({})

        for h_0_index in range(self.prob_y1_h_b[0].columns.shape[0] - 1):
            for b_0_index in range(self.prob_y1_h_b[0].index.shape[0] - 1):
                for h_prime_0_index in range(
                    h_0_index, self.prob_y1_h_b[0].columns.shape[0]
                ):
                    for b_prime_0_index in range(
                        b_0_index, self.prob_y1_h_b[0].index.shape[0]
                    ):
                        for h_1_index in range(h_0_index, h_0_index + 1):
                            for h_prime_1_index in range(
                                h_prime_0_index, h_prime_0_index + 1
                            ):
                                for b_1_index in range(b_0_index, b_0_index + 1):
                                    for b_prime_1_index in range(
                                        b_prime_0_index, b_prime_0_index + 1
                                    ):
                                        num_summants += 1
                                        # check misalignment of the pair of cells if enough datapoints in each cells
                                        if (
                                            self.mass_y1_h_b[0].iat[
                                                b_prime_0_index, h_prime_0_index
                                            ]
                                            + self.mass_y1_h_b[1].iat[
                                                b_prime_1_index, h_prime_1_index
                                            ]
                                            >= self.min_datapoints
                                            and self.mass_y1_h_b[0].iat[
                                                b_0_index, h_0_index
                                            ]
                                            + self.mass_y1_h_b[1].iat[
                                                b_1_index, h_1_index
                                            ]
                                            >= self.min_datapoints
                                        ):
                                            alignment = max(
                                                0.0,
                                                self.df_cell_prob.iat[
                                                    b_0_index, h_0_index
                                                ]
                                                - self.df_cell_prob.iat[
                                                    b_prime_0_index, h_prime_0_index
                                                ],
                                            )

                                            alignment2 = max(
                                                0.0,
                                                (
                                                    self.prob_y1_h_b[0].iat[
                                                        b_0_index, h_0_index
                                                    ]
                                                    * self.mass_y1_h_b[0].iat[
                                                        b_0_index, h_0_index
                                                    ]
                                                    / (
                                                        self.mass_y1_h_b[0].iat[
                                                            b_0_index, h_0_index
                                                        ]
                                                        + self.mass_y1_h_b[1].iat[
                                                            b_0_index, h_0_index
                                                        ]
                                                    )
                                                    + self.prob_y1_h_b[1].iat[
                                                        b_1_index, h_1_index
                                                    ]
                                                    * self.mass_y1_h_b[1].iat[
                                                        b_1_index, h_1_index
                                                    ]
                                                    / (
                                                        self.mass_y1_h_b[0].iat[
                                                            b_1_index, h_1_index
                                                        ]
                                                        + self.mass_y1_h_b[1].iat[
                                                            b_1_index, h_1_index
                                                        ]
                                                    )
                                                )
                                                - (
                                                    self.prob_y1_h_b[0].iat[
                                                        b_prime_0_index, h_prime_0_index
                                                    ]
                                                    * self.mass_y1_h_b[0].iat[
                                                        b_prime_0_index, h_prime_0_index
                                                    ]
                                                    / (
                                                        self.mass_y1_h_b[0].iat[
                                                            b_prime_0_index,
                                                            h_prime_0_index,
                                                        ]
                                                        + self.mass_y1_h_b[1].iat[
                                                            b_prime_0_index,
                                                            h_prime_0_index,
                                                        ]
                                                    )
                                                    + self.prob_y1_h_b[1].iat[
                                                        b_prime_1_index, h_prime_1_index
                                                    ]
                                                    * self.mass_y1_h_b[1].iat[
                                                        b_prime_1_index, h_prime_1_index
                                                    ]
                                                    / (
                                                        self.mass_y1_h_b[0].iat[
                                                            b_prime_1_index,
                                                            h_prime_1_index,
                                                        ]
                                                        + self.mass_y1_h_b[1].iat[
                                                            b_prime_1_index,
                                                            h_prime_1_index,
                                                        ]
                                                    )
                                                ),
                                            )

                                            max_aligment = max(max_aligment, alignment)
                                            if alignment > 0.0:
                                                sum_aligment += alignment
                                                disaligned_cells |= {
                                                    (
                                                        self.prob_y1_h_b[0].index[
                                                            b_0_index
                                                        ],
                                                        self.prob_y1_h_b[0].columns[
                                                            h_0_index
                                                        ],
                                                        self.prob_y1_h_b[1].index[
                                                            b_1_index
                                                        ],
                                                        self.prob_y1_h_b[1].columns[
                                                            h_1_index
                                                        ],
                                                    ),
                                                    (
                                                        self.prob_y1_h_b[0].index[
                                                            b_prime_0_index
                                                        ],
                                                        self.prob_y1_h_b[0].columns[
                                                            h_prime_0_index
                                                        ],
                                                        self.prob_y1_h_b[1].index[
                                                            b_prime_1_index
                                                        ],
                                                        self.prob_y1_h_b[1].columns[
                                                            h_prime_1_index
                                                        ],
                                                    ),
                                                }

                                            # case 1
                                        if (
                                            self.mass_y1_h_b[0].iat[
                                                b_prime_0_index, h_prime_0_index
                                            ]
                                            >= self.min_datapoints // 2
                                            and self.mass_y1_h_b[1].iat[
                                                b_prime_1_index, h_prime_1_index
                                            ]
                                            >= self.min_datapoints // 2
                                            and self.mass_y1_h_b[0].iat[
                                                b_0_index, h_0_index
                                            ]
                                            >= self.min_datapoints // 2
                                            and self.mass_y1_h_b[1].iat[
                                                b_1_index, h_1_index
                                            ]
                                            >= self.min_datapoints // 2
                                        ):
                                            alignment_s = abs(
                                                self.prob_y1_h_b[1].iat[
                                                    b_1_index, h_1_index
                                                ]
                                                - self.prob_y1_h_b[0].iat[
                                                    b_0_index, h_0_index
                                                ]
                                            ) + abs(
                                                -self.prob_y1_h_b[1].iat[
                                                    b_prime_1_index,
                                                    h_prime_1_index,
                                                ]
                                                + self.prob_y1_h_b[0].iat[
                                                    b_prime_0_index,
                                                    h_prime_0_index,
                                                ]
                                            )
                                            max_aligment_s = max(
                                                max_aligment_s, alignment_s
                                            )
                                            if alignment_s > 0:
                                                sum_aligment_s += alignment_s
                                                disaligned_cells_s |= {
                                                    (
                                                        self.prob_y1_h_b[0].index[
                                                            b_0_index
                                                        ],
                                                        self.prob_y1_h_b[0].columns[
                                                            h_0_index
                                                        ],
                                                        self.prob_y1_h_b[1].index[
                                                            b_1_index
                                                        ],
                                                        self.prob_y1_h_b[1].columns[
                                                            h_1_index
                                                        ],
                                                    ),
                                                    (
                                                        self.prob_y1_h_b[0].index[
                                                            b_prime_0_index
                                                        ],
                                                        self.prob_y1_h_b[0].columns[
                                                            h_prime_0_index
                                                        ],
                                                        self.prob_y1_h_b[1].index[
                                                            b_prime_1_index
                                                        ],
                                                        self.prob_y1_h_b[1].columns[
                                                            h_prime_1_index
                                                        ],
                                                    ),
                                                }

        if num_summants > 0:
            avg_alignment = sum_aligment / num_summants
            avg_alignment_s = sum_aligment_s / num_summants
        else:
            avg_alignment = 0.0
            avg_alignment_s = 0.0

        self.disaligned_cells = disaligned_cells
        self.disaligned_cells_s = disaligned_cells_s

        mae = max_aligment
        eae = avg_alignment

        mae_s = max_aligment_s
        eae_s = avg_alignment_s

        return (eae, mae, eae_s, mae_s)

    # computes expected and maximum calibration error
    def check_calibration(self):
        df = self.task_data.copy()
        df_female = df[df["gender"] == "female"]
        df_male = df[df["gender"] == "male"]

        prob_true_female, prob_pred_female = calibration_curve(
            df_female["y"], df_female["b"], n_bins=self.b_bins
        )

        prob_true_male, prob_pred_male = calibration_curve(
            df_male["y"], df_male["b"], n_bins=self.b_bins
        )

        abs_diff = pd.DataFrame(
            data={
                "b_mass": (prob_true_female - prob_true_male)
                - (prob_pred_female - prob_pred_male)
            }
        )
        abs_diff["b_mass"] = abs_diff["b_mass"].abs()

        # compute maximum calibration error (MCE)
        mce = abs_diff["b_mass"].max()

        # compute expected calibration error (ECE)
        df_density = [
            self.mass_y1_h_b[i] / self.mass_y1_h_b[i].sum().sum()
            for i in range(len(self.mass_y1_h_b))
        ]
        b_bin_mass = [df_density[i].sum(axis=1) for i in range(len(df_density))]

        ece = abs(
            (b_bin_mass[0] * prob_true_female - b_bin_mass[1] * prob_true_male)
            - (b_bin_mass[0] * prob_pred_female - b_bin_mass[1] * prob_pred_male)
        )
        ece = ece.sum(axis=0)

        return (ece, mce)

    def run_experiment(self):

        # compute expected and maximum alignment error
        eae, mae, eae0, mae0 = self.check_alignment()

        # # compute calibration measures ECE and MCE
        # ece, mce = self.check_calibration()

        # compute ROC AUC
        roc_auc_h = roc_auc_score(self.task_data["y"], self.task_data["h"])
        roc_auc_hAI = roc_auc_score(self.task_data["y"], self.task_data["h+AI"])
        roc_auc_b = roc_auc_score(self.task_data["y"], self.task_data["b"])

        roc_auc_h_disp = abs(
            roc_auc_score(
                self.task_data[self.task_data["gender"] == "female"]["y"],
                self.task_data[self.task_data["gender"] == "female"]["h"],
            )
            - roc_auc_score(
                self.task_data[self.task_data["gender"] == "male"]["y"],
                self.task_data[self.task_data["gender"] == "male"]["h"],
            )
        )
        roc_auc_hAI_disp = abs(
            roc_auc_score(
                self.task_data[self.task_data["gender"] == "female"]["y"],
                self.task_data[self.task_data["gender"] == "female"]["h+AI"],
            )
            - roc_auc_score(
                self.task_data[self.task_data["gender"] == "male"]["y"],
                self.task_data[self.task_data["gender"] == "male"]["h+AI"],
            )
        )
        roc_auc_b_disp = abs(
            roc_auc_score(
                self.task_data[self.task_data["gender"] == "female"]["y"],
                self.task_data[self.task_data["gender"] == "female"]["b"],
            )
            - roc_auc_score(
                self.task_data[self.task_data["gender"] == "male"]["y"],
                self.task_data[self.task_data["gender"] == "male"]["b"],
            )
        )

        acc_hAI = accuracy_score(
            self.task_data["y"],
            (self.task_data["h+AI"]).astype(int),
        )

        acc_h = accuracy_score(
            self.task_data["y"], (self.task_data["h"] >= 0.5).astype(int)
        )

        acc_b = accuracy_score(
            self.task_data["y"], (self.task_data["b"] >= 0.5).astype(int)
        )

        acc_h_disp = accuracy_score(
            self.task_data[self.task_data["gender"] == "female"]["y"],
            (self.task_data[self.task_data["gender"] == "female"]["h"] >= 0.5).astype(
                int
            ),
        ) - accuracy_score(
            self.task_data[self.task_data["gender"] == "male"]["y"],
            (self.task_data[self.task_data["gender"] == "male"]["h"] >= 0.5).astype(
                int
            ),
        )

        acc_hAI_disp = accuracy_score(
            self.task_data[self.task_data["gender"] == "female"]["y"],
            (self.task_data[self.task_data["gender"] == "female"]["h+AI"]),
        ) - accuracy_score(
            self.task_data[self.task_data["gender"] == "male"]["y"],
            (self.task_data[self.task_data["gender"] == "male"]["h+AI"]),
        )

        acc_b_disp = accuracy_score(
            self.task_data[self.task_data["gender"] == "female"]["y"],
            (self.task_data[self.task_data["gender"] == "female"]["b"] >= 0.5).astype(
                int
            ),
        ) - accuracy_score(
            self.task_data[self.task_data["gender"] == "male"]["y"],
            (self.task_data[self.task_data["gender"] == "male"]["b"] >= 0.5).astype(
                int
            ),
        )

        # compute TP
        confusion_matrix_h_female = confusion_matrix(
            self.task_data[self.task_data["gender"] == "female"]["y"],
            (self.task_data[self.task_data["gender"] == "female"]["h"]).astype(int),
        )
        confusion_matrix_h_male = confusion_matrix(
            self.task_data[self.task_data["gender"] == "male"]["y"],
            (self.task_data[self.task_data["gender"] == "male"]["h"]).astype(int),
        )
        positive_h = (confusion_matrix_h_male[1, 1]) / (
            confusion_matrix_h_male[1, 0] + confusion_matrix_h_male[1, 1]
        ) - (confusion_matrix_h_female[1, 1]) / (
            confusion_matrix_h_female[1, 0] + confusion_matrix_h_female[1, 1]
        )

        confusion_matrix_h_female = confusion_matrix(
            self.task_data[self.task_data["gender"] == "female"]["y"],
            (
                self.task_data[self.task_data["gender"] == "female"]["h+AI"] >= 0.5
            ).astype(int),
        )
        confusion_matrix_h_male = confusion_matrix(
            self.task_data[self.task_data["gender"] == "male"]["y"],
            (self.task_data[self.task_data["gender"] == "male"]["h+AI"] >= 0.5).astype(
                int
            ),
        )

        # positive_hAI = (confusion_matrix_h_male[1, 1]) / (
        #     confusion_matrix_h_male[1, 0] + confusion_matrix_h_male[1, 1]
        # ) - (confusion_matrix_h_female[1, 1]) / (
        #     confusion_matrix_h_female[1, 0] + confusion_matrix_h_female[1, 1]
        # )
        positive_hAI_TP = (confusion_matrix_h_male[1, 1]) / (
            confusion_matrix_h_male[1, 0] + confusion_matrix_h_male[1, 1]
        ) - (confusion_matrix_h_female[1, 1]) / (
            confusion_matrix_h_female[1, 0] + confusion_matrix_h_female[1, 1]
        )
        positive_hAI_FP = (confusion_matrix_h_male[0, 1]) / (
            confusion_matrix_h_male[0, 0] + confusion_matrix_h_male[0, 1]
        ) - (confusion_matrix_h_female[0, 1]) / (
            confusion_matrix_h_female[0, 0] + confusion_matrix_h_female[1, 1]
        )
        positive_hAI = (confusion_matrix_h_male[1, 1]) / (
            confusion_matrix_h_male[1, 0] + confusion_matrix_h_male[1, 1]
        ) - (confusion_matrix_h_female[1, 1]) / (
            confusion_matrix_h_female[1, 0] + confusion_matrix_h_female[1, 1]
        )
        confusion_matrix_h_female = confusion_matrix(
            self.task_data[self.task_data["gender"] == "female"]["y"],
            (self.task_data[self.task_data["gender"] == "female"]["b"] >= 0.5).astype(
                int
            ),
        )
        confusion_matrix_h_male = confusion_matrix(
            self.task_data[self.task_data["gender"] == "male"]["y"],
            (self.task_data[self.task_data["gender"] == "male"]["b"] >= 0.5).astype(
                int
            ),
        )

        positive_b = (confusion_matrix_h_male[1, 1]) / (
            confusion_matrix_h_male[1, 0] + confusion_matrix_h_male[1, 1]
        ) - (confusion_matrix_h_female[1, 1]) / (
            confusion_matrix_h_female[1, 0] + confusion_matrix_h_female[1, 1]
        )

        # bias 2
        bias_h = (
            self.task_data[
                (self.task_data["gender"] == "male") & (self.task_data["y"] == 1)
            ]["h"].mean()
            - self.task_data[
                (self.task_data["gender"] == "female") & (self.task_data["y"] == 1)
            ]["h"].mean()
        )

        bias_b = (
            self.task_data[
                (self.task_data["gender"] == "male") & (self.task_data["y"] == 1)
            ]["b"].mean()
            - self.task_data[
                (self.task_data["gender"] == "female") & (self.task_data["y"] == 1)
            ]["b"].mean()
        )

        bias_hai = (
            self.task_data[
                (self.task_data["gender"] == "male") & (self.task_data["y"] == 1)
            ]["h+AI"].mean()
            - self.task_data[
                (self.task_data["gender"] == "female") & (self.task_data["y"] == 1)
            ]["h+AI"].mean()
        )

        dict_metrics = {
            "EAE": [round(eae, 4)],
            "MAE": [round(mae, 4)],
            "EAE_S": [round(eae0, 4)],
            "MAE_S": [round(mae0, 4)],
            # "ECE": [ece],
            # "MCE": [mce],
            "acc_b": [acc_b],
            "acc_h": [acc_h],
            "acc_h+AI": [acc_hAI],
            "acc_b_disp": [acc_b_disp],
            "acc_h_disp": [acc_h_disp],
            "acc_h+AI_disp": [acc_hAI_disp],
            # "positive_b": [positive_b],
            # "positive_h": [positive_h],
            # "positive_h+AI": [positive_hAI],
            # "bias_b": [bias_b],
            # "bias_h": [bias_h],
            # "bias_h+AI": [bias_hai],
        }
        self.df_metrics = pd.DataFrame(dict_metrics, index=[self.task_name]).round(4)

        # all_hatched = []
        # for h_0 in range(self.prob_y1_h_b[0].columns.shape[0]):
        #     for b_0 in range(self.prob_y1_h_b[0].index.shape[0]):
        #         for h_1 in range(self.prob_y1_h_b[0].columns.shape[0]):
        #             for b_1 in range(self.prob_y1_h_b[0].index.shape[0]):
        #                 if (
        #                     self.prob_y1_h_b[0].index[b_0],
        #                     self.prob_y1_h_b[0].columns[h_0],
        #                     self.prob_y1_h_b[1].index[b_1],
        #                     self.prob_y1_h_b[1].columns[h_1],
        #                 ) in self.disaligned_cells:
        #                     all_hatched.append("//" * 3)
        #                 else:
        #                     all_hatched.append("")

        # fairness_alignment_barplot(
        #     self.task_data, all_hatched, self.min_datapoints, self.task_name
        # )
        # confidence_change_barplot(self.task_data, all_hatched, self.task_name)
        # plot_histogram_cells(self.task_data, self.task_name)
        # plot_roc(self.task_data, self.task_name)

    def get_task_data(self):
        return self.task_data

    def get_metrics(self):
        return self.df_metrics


# plots ROC curve
def plot_roc(data, task_name):
    fig, ax = plt.subplots()
    fpr, tpr, th = roc_curve(data["y"], data["b"])
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot(ax=ax, linewidth=3)

    fpr, tpr, th = roc_curve(data["y"], data["h"])
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot(ax=ax, linewidth=3)

    fpr, tpr, th = roc_curve(data["y"], data["h+AI"])
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot(ax=ax, linewidth=3)
    ax.set_xlabel("False Positive Rate", fontsize=18)
    ax.set_ylabel("True Positive Rate", fontsize=18)
    ax.tick_params(labelsize=18)
    plt.legend(
        [r"$\pi_B$", r"$\pi_H$", r"$\pi_{H_\mathrm{+AI}}$"],
        title="Decision Policies",
        title_fontsize=18,
        prop={"size": 18},
        loc="lower right",
    )
    plt.tight_layout()
    plt.savefig(path + "/roc/roc_" + task_name + ".pdf")
    plt.close()


# barplot of cell probabilities P(Y=1 | cell=(h,b))
def alignment_barplot(df, bar_hatches, min_cell_mass, name):

    #
    h_conf_bin_limits = [
        df[df["gender"] == "female"]["h_bin"].unique(),
        df[df["gender"] == "male"]["h_bin"].unique(),
    ]
    legend_array = [
        "'low'- female: [0.0, "
        + str(h_conf_bin_limits[0][0])
        + "]"
        + "male: [0.0, "
        + str(h_conf_bin_limits[1][0])
        + "]"
    ]
    legend_array += [
        "'mid'- female: ("
        + str(h_conf_bin_limits[0][0])
        + ", "
        + str(h_conf_bin_limits[0][1])
        + "]"
        + "male: ("
        + str(h_conf_bin_limits[1][0])
        + ", "
        + str(h_conf_bin_limits[1][1])
        + "]"
    ]
    legend_array += [
        "'high'- female: ("
        + str(h_conf_bin_limits[0][1])
        + ", "
        + str(h_conf_bin_limits[0][2])
        + "]"
        + "male: ("
        + str(h_conf_bin_limits[1][1])
        + ", "
        + str(h_conf_bin_limits[1][2])
        + "]"
    ]

    # plot figures
    custom_params = {
        "axes.edgecolor": "lightgray"
    }  # "axes.spines.right": False, "axes.spines.top": False,
    sns.set(context="paper", style="ticks", color_codes=True, rc=custom_params)

    # plot all bins with errorbar
    #
    # Calculate mean and confidence interval for female data
    female_data = (
        df[df["gender"] == "female"]
        .groupby(["b_bin", "h_bin_level"])
        .agg(t_mean=("h+AI", np.nanmean), y_mean=("y", np.nanmean))
        .reset_index()
    )

    # Calculate mean and confidence interval for male data
    male_data = (
        df[df["gender"] == "male"]
        .groupby(["b_bin", "h_bin_level"])
        .agg(t_mean=("h+AI", np.nanmean), y_mean=("y", np.nanmean))
        .reset_index()
    )

    # # Add rank column to female_data and male_data, ensuring same rank for same b_bin value
    # female_data["h_rank"] = female_data["h_bin"].rank(method="dense").astype(int)
    # male_data["h_rank"] = male_data["h_bin"].rank(method="dense").astype(int)

    # Merge data
    merged_data = pd.merge(
        female_data,
        male_data,
        on=["b_bin", "h_bin_level"],
        suffixes=("_female", "_male"),
    )

    # 添加辅助列以进行笛卡尔积
    female_data["key"] = 1
    male_data["key"] = 1

    # 笛卡尔积合并
    merged_data_decare = pd.merge(
        female_data, male_data, on="key", suffixes=("_female", "_male")
    ).drop("key", axis=1)

    filtered_data = merged_data_decare[
        (merged_data_decare["h_bin_level_male"] == 2)
        & (merged_data_decare["h_bin_level_female"] == 0)
    ].reset_index()

    # Calculate the difference in corresponding data
    merged_data["y_diff"] = merged_data["y_mean_female"] - merged_data["y_mean_male"]
    merged_data["t_diff"] = merged_data["t_mean_female"] - merged_data["t_mean_male"]
    # Plot the results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    bar1 = sns.barplot(
        data=merged_data,
        x="b_bin",
        y="y_diff",
        hue="h_bin_level",
        palette="colorblind",
        # estimator=np.nanmean,
        errorbar=("ci", 90),  # Add 90% confidence interval error bars
        errwidth=0.12,
        capsize=0.12,
    )
    bar2 = sns.barplot(
        data=merged_data,
        x="b_bin",
        y="t_diff",
        hue="h_bin_level",
        palette="colorblind",
        # estimator=np.nanmean,
        errorbar=("ci", 90),  # Add 90% confidence interval error bars
        errwidth=0.12,
        capsize=0.12,
    )
    #
    # bar = sns.barplot(
    #     x="b_bin",
    #     y="y",
    #     hue="h_bin",
    #     estimator=np.nanmean,
    #     errorbar=("ci", 90),
    #     errwidth=0.12,
    #     capsize=0.12,
    #     data=df[df["gender"] == "female"],
    #     palette="colorblind",
    #     ax=ax,
    # )
    # bar = sns.barplot(
    #     x="b_bin",
    #     y="y",
    #     hue="h_bin",
    #     estimator=np.nanmean,
    #     errorbar=("ci", 90),
    #     errwidth=0.12,
    #     capsize=0.12,
    #     data=df[df["gender"] == "male"],
    #     palette="colorblind",
    #     ax=ax,
    # )
    # set hatch to bars with disalignment (only considers disalignment on bins with mass >= min_bin_mass_prob)
    if len(bar_hatches) > 0:
        for i, b in enumerate(bar1.patches):
            b.set_hatch(bar_hatches[i])
    if len(bar_hatches) > 0:
        for i, b in enumerate(bar2.patches):
            b.set_hatch(bar_hatches[i])

    ax1.set_xlabel(r"Model Confidence, $b$", fontsize=18)
    ax1.set_ylabel(r"$P(Y=1 \mid (X,Y) \in \mathcal{S}_{h,\lambda(b)})$", fontsize=18)
    ax1.tick_params(labelsize=18)

    ax2.set_xlabel(r"Model Confidence, $b$", fontsize=18)
    ax2.set_ylabel(r"$P(Y=1 \mid (X,Y) \in \mathcal{S}_{h,\lambda(b)})$", fontsize=18)
    ax2.tick_params(labelsize=18)
    # ax.set_ylim(-0.02, 1.05)
    # change legend
    h, _ = bar1.get_legend_handles_labels()
    bar1.legend(
        h,
        legend_array,
        title=r"Human Confidence, $h$",
        loc="upper left",
        prop={"size": 18},
        title_fontsize=18,
    )

    plt.tight_layout()
    plt.savefig(path + "/barplot/alignment_" + name + ".pdf")
    plt.close()


def fairness_alignment_barplot(df, bar_hatches, min_cell_mass, name):
    #
    b_conf_bin_limits = np.sort(df[df["gender"] == "male"]["b_bin"].unique())
    h_conf_bin_limits = np.sort(df[df["gender"] == "female"]["h_bin_level"].unique())
    legend_array = []
    for i in range(len(b_conf_bin_limits)):
        latex_str = r"$\lambda(a_1)$="
        legend_array.append(latex_str + str(b_conf_bin_limits[i]))

    # plot figures
    custom_params = {
        "axes.edgecolor": "lightgray"
    }  # "axes.spines.right": False, "axes.spines.top": False,
    sns.set(context="paper", style="ticks", color_codes=True, rc=custom_params)
    # plot all bins with errorbar
    #
    # Calculate mean and confidence interval for female data
    female_data = (
        df[df["gender"] == "female"]
        .groupby(["b_bin", "h_bin_level"])
        .agg(
            t_mean=("h+AI", lambda x: np.nanmean(x[df.loc[x.index, "y"] == 1])),
            y_mean=("y", np.nanmean),
            y_std=("y", np.nanstd),
            h_bin_mean=("h_bin", np.nanmean),
            t_y=(
                "h+AI",
                lambda x: np.mean((x > 0.5).astype(int) == df.loc[x.index, "y"]),
            ),
        )
        .reset_index()
    )

    # Calculate mean and confidence interval for male data
    male_data = (
        df[df["gender"] == "male"]
        .groupby(["b_bin", "h_bin_level"])
        .agg(
            t_mean=("h+AI", lambda x: np.nanmean(x[df.loc[x.index, "y"] == 1])),
            y_mean=("y", np.nanmean),
            y_std=("y", np.nanstd),
            h_bin_mean=("h_bin", np.nanmean),
            t_y=(
                "h+AI",
                lambda x: np.mean((x > 0.5).astype(int) == df.loc[x.index, "y"]),
            ),
        )
        .reset_index()
    )

    # 添加辅助列以进行笛卡尔积
    female_data["key"] = 1
    male_data["key"] = 1

    # 笛卡尔积合并
    merged_data_decare = pd.merge(
        female_data, male_data, on="key", suffixes=("_female", "_male")
    ).drop("key", axis=1)

    # Compute covariance
    # Ensure that y values for both genders are paired correctly
    male_y_values = merged_data_decare["y_mean_male"]
    female_y_values = merged_data_decare["y_mean_female"]

    mean_male_y = np.nanmean(male_y_values)
    mean_female_y = np.nanmean(female_y_values)

    # Compute covariance
    cov = np.nanmean((male_y_values - mean_male_y) * (female_y_values - mean_female_y))

    filtered_data = merged_data_decare
    # selected_row = filtered_data.iloc[0]
    # h_diff = selected_row["h_bin_mean_male"] - selected_row["h_bin_mean_female"]
    # Calculate the difference in corresponding data
    filtered_data["t_y_diff"] = filtered_data["t_y_male"] - filtered_data["t_y_female"]
    # filtered_data["y_diff_std"] = np.sqrt(
    #     filtered_data["y_std_male"] ** 2 + filtered_data["y_std_female"] ** 2 - 2 * cov
    # )

    # filtered_data["t_diff"] = (
    #     filtered_data["t_mean_male"] - filtered_data["t_mean_female"]
    # )

    # Plot the results
    fig, ax = plt.subplots(2, 1, figsize=(16, 12))

    bar1 = sns.barplot(
        data=filtered_data,
        x="b_bin_female",
        y="t_y_female",
        estimator=np.nanmean,
        hue="h_bin_level_female",
        palette="colorblind",
        # estimator=np.nanmean,
        errorbar=("ci", 90),  # Add 90% confidence interval error bars
        errwidth=0.08,
        capsize=0.12,
        ax=ax[0],
    )
    bar2 = sns.barplot(
        data=filtered_data,
        x="b_bin_male",
        y="t_y_male",
        estimator=np.nanmean,
        hue="h_bin_level_male",
        palette="colorblind",
        # estimator=np.nanmean,
        errorbar=("ci", 90),  # Add 90% confidence interval error bars
        errwidth=0.08,
        capsize=0.12,
        ax=ax[1],
    )
    # bar2 = sns.barplot(
    #     data=filtered_data,
    #     x="b_bin_female",
    #     y="t_y_male",
    #     estimator=np.nanmean,
    #     hue="b_bin_male",
    #     palette="colorblind",
    #     # estimator=np.nanmean,
    #     errorbar=("ci", 90),  # Add 90% confidence interval error bars
    #     errwidth=0.08,
    #     capsize=0.12,
    #     ax=ax[1],
    # )
    for i, line in enumerate(bar1.get_lines()):
        line.set_alpha(1)
        line.set_linewidth(1)
    for i, line in enumerate(bar2.get_lines()):
        line.set_alpha(1)
        line.set_linewidth(1)
    # 添加误差条
    # for patch, std in zip(bar1.patches, filtered_data["y_diff_std"]):
    #     height = patch.get_height()
    #     bar1.errorbar(
    #         x=patch.get_x() + patch.get_width() / 2,
    #         y=height,
    #         yerr=std,
    #         fmt="none",
    #         capsize=0.12,
    #         color="black",
    #     )
    # bar2 = sns.barplot(
    #     data=filtered_data,
    #     x="b_bin_female",
    #     y="t_diff",
    #     hue="b_bin_male",
    #     palette="colorblind",
    #     # estimator=np.nanmean,
    #     errorbar=("ci", 90),  # Add 90% confidence interval error bars
    #     errwidth=0.12,
    #     capsize=0.12,
    #     ax=ax2,
    # )
    # set hatch to bars with disalignment (only considers disalignment on bins with mass >= min_bin_mass_prob)
    if len(bar_hatches) > 0:
        for i, b in enumerate(bar1.patches):
            b.set_hatch(bar_hatches[i])
    if len(bar_hatches) > 0:
        for i, b in enumerate(bar2.patches):
            b.set_hatch(bar_hatches[i])

    ax[0].set_xlabel(r"$AI confidence$", fontsize=14)
    # ax1.set_ylabel(r" $a_1$", fontsize=18)
    ax[0].set_ylabel(
        r"$Utility$",
        fontsize=14,
    )
    ax[0].tick_params(labelsize=14)
    ax[1].set_xlabel(r"$AI confidence$", fontsize=14)
    # ax1.set_ylabel(r" $a_1$", fontsize=18)
    ax[1].set_ylabel(
        r"$Utility$",
        fontsize=14,
    )
    ax[1].tick_params(labelsize=14)
    # ax2.set_xlabel(r"Female AI Confidence, $b$", fontsize=18)
    # ax2.set_ylabel(
    #     r"$P(\pi=1 \mid (X,Y) \in \mathcal{S}_{s=female, h,\lambda(b)})-P(\pi=1 \mid (X,Y) \in \mathcal{S}_{s=male, h,\lambda(b)})$",
    #     fontsize=18,
    # )
    # ax2.tick_params(labelsize=18)
    # ax2.legend().remove()
    # ax.set_ylim(-0.02, 1.05)
    # change legend
    h, _ = bar1.get_legend_handles_labels()
    bar1.legend(
        h,
        legend_array,
        title=r"Human Confidence",
        loc="upper right",
        # bbox_to_anchor=(1, 0.5),
        prop={"size": 12},
    )
    h, _ = bar2.get_legend_handles_labels()
    bar2.legend(
        h,
        legend_array,
        title=r"Human Confidence",
        loc="upper right",
        # bbox_to_anchor=(1, 0.5),
        prop={"size": 12},
    )

    # plt.tight_layout()
    plt.savefig(path + "/barplot/fairness_alignment_" + name + ".pdf")
    plt.savefig(path + "/barplot/fairness_alignment_" + name + ".svg")
    plt.close()


# barplot of cell expectation of confidence change E(h_+AI - h | cell=(h,b))
def confidence_change_barplot(df, bar_hatches, name):

    df["confidence_change"] = df["h+AI"] - df["h"]
    # plot figures
    custom_params = {
        "axes.edgecolor": "lightgray"
    }  # "axes.spines.right": False, "axes.spines.top": False,
    sns.set(context="paper", style="ticks", color_codes=True, rc=custom_params)
    fig, ax = plt.subplots(figsize=(10, 5))
    # plot all bins with errorbar
    bar = sns.barplot(
        x="b_bin",
        y="confidence_change",
        hue="h_bin",
        estimator=np.nanmean,
        errorbar=("ci", 90),
        errwidth=0.12,
        capsize=0.12,
        data=df,
        palette="colorblind",
        ax=ax,
    )

    # set hatch to bars with disalignment (only considers disalignment on bins with mass >= min_bin_mass_prob)
    if len(bar_hatches) > 0:
        for i, b in enumerate(bar.patches):
            b.set_hatch(bar_hatches[i])

    h_conf_bin_limits = df["h_bin"].unique()
    legend_array = ["'low': [0.0, " + str(h_conf_bin_limits[0]) + "]"]
    legend_array += [
        "'mid': (" + str(h_conf_bin_limits[0]) + ", " + str(h_conf_bin_limits[1]) + "]"
    ]
    legend_array += [
        "'high': (" + str(h_conf_bin_limits[1]) + ", " + str(h_conf_bin_limits[2]) + "]"
    ]

    # plot only large bins with errorbar
    ax.set_xlabel(r"Model Confidence, $b$", fontsize=18)
    ax.set_ylabel(
        r"$E[ h_{\mathrm{+AI}} - h \mid (X,Y) \in \mathcal{S}_{h,\lambda(b)}]$",
        fontsize=18,
    )
    ax.tick_params(labelsize=18)
    # change legend
    h, _ = bar.get_legend_handles_labels()
    bar.legend(
        h,
        legend_array,
        title=r"Human Confidence, $h$",
        loc="lower right",
        prop={"size": 18},
        title_fontsize=18,
    )
    plt.tight_layout()
    plt.savefig(path + "/barplot/confidence_change_" + name + ".pdf")
    plt.close()


# histogram of cell mass
def plot_histogram_cells(df, name):

    # 2d histogram
    custom_params = {"axes.edgecolor": "lightgray"}
    sns.set(context="paper", style="ticks", color_codes=True, rc=custom_params)
    fig, ax = plt.subplots(figsize=(10, 5))

    mass_y1_h_b = df[["b_bin", "h_bin", "y"]]
    mass_y1_h_b = mass_y1_h_b.groupby(["h_bin", "b_bin"]).count().unstack()
    df_reverse = mass_y1_h_b.reindex(index=mass_y1_h_b.index[::-1])

    bar = sns.heatmap(df_reverse, cbar=True, cmap="crest")
    cbar = bar.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)
    ax.set_ylabel(r"Human Confidence $h$", fontsize=16)
    ax.set_xlabel(r"Model Confidence $b$", fontsize=16)
    ax.tick_params(labelsize=16)

    plt.tight_layout()
    plt.savefig(path + "/hist/hist2d_" + name + ".pdf")
    plt.close()


def main(method):
    before_calibration_results = []
    after_calibration_results = []
    for k in range(100):
        print(k, "-th iterations:\n")
        exp_art = Experiment("art", h_bins=3, b_bins=8, random_state=k)
        exp_sarcasm = Experiment("sarcasm", h_bins=3, b_bins=8, random_state=k)
        exp_cities = Experiment("cities", h_bins=3, b_bins=8, random_state=k)
        exp_census = Experiment("census", h_bins=3, b_bins=8, random_state=k)

        df_results = exp_art.get_metrics()
        df_results = pd.concat([df_results, exp_sarcasm.get_metrics()])
        df_results = pd.concat([df_results, exp_cities.get_metrics()])
        df_results = pd.concat([df_results, exp_census.get_metrics()])

        before_calibration_results.append(df_results)
        print("Before Calibration:\n", df_results)

         # Setting the method to use for calibration
        if method == 'fair':
            # Apply fair_aware_multicalibration
            w1 = 1
            w2 = 0
            exp_art.fair_aware_multicalibration(alpha=0.0001, w1=w1, w2=w2)
            exp_sarcasm.fair_aware_multicalibration(alpha=0.0001, w1=w1, w2=w2)
            exp_cities.fair_aware_multicalibration(alpha=0.0001, w1=w1, w2=w2)
            exp_census.fair_aware_multicalibration(alpha=0.0001, w1=w1, w2=w2)
        else:
            # Apply regular multicalibration
            exp_art.multicalibration(alpha=0.0001)
            exp_sarcasm.multicalibration(alpha=0.0001)
            exp_cities.multicalibration(alpha=0.0001)
            exp_census.multicalibration(alpha=0.0001)

        # Compute the cell probability matrix and run the experiment
        exp_art.compute_cell_prob_matrix()
        exp_art.run_experiment()

        exp_sarcasm.compute_cell_prob_matrix()
        exp_sarcasm.run_experiment()

        exp_cities.compute_cell_prob_matrix()
        exp_cities.run_experiment()

        exp_census.compute_cell_prob_matrix()
        exp_census.run_experiment()

        # Collect the results after calibration
        df_results = exp_art.get_metrics()
        df_results = pd.concat([df_results, exp_sarcasm.get_metrics()])
        df_results = pd.concat([df_results, exp_cities.get_metrics()])
        df_results = pd.concat([df_results, exp_census.get_metrics()])

        after_calibration_results.append(df_results)
        print("After Calibration:\n", df_results)

    # Save results to files with conditional naming
    before_file_name = "before_fair_calibration_results.pkl" if method == 'fair' else "before_calibration_results.pkl"
    after_file_name = "after_fair_calibration_results.pkl" if method == 'fair' else "after_calibration_results.pkl"

    with open(before_file_name, "wb") as f:
        pickle.dump(before_calibration_results, f)

    with open(after_file_name, "wb") as f:
        pickle.dump(after_calibration_results, f)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run experiments with calibration methods")
    parser.add_argument('--method', choices=['fair', 'standard'], default='standard', 
                        help="Specify the calibration method: 'fair' for group-level multicalibration, 'standard' for multicalibration")
    args = parser.parse_args()

    # Run the main function with the selected method
    main(args.method)
