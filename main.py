import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import model_selection, preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay


universal_no_outliers = r'C:\Users\Esau\PycharmProjects\PersonalProject\universal_no_outliers.csv'
universal = r'C:\Users\Esau\PycharmProjects\PersonalProject\universal_full.csv'
english_no_outliers = r'C:\Users\Esau\PycharmProjects\PersonalProject\english_no_outliers.csv'
english = r'C:\Users\Esau\PycharmProjects\PersonalProject\english_full.csv'


def orgs_to_bots(df):
    df = df.replace({'label': 2}, {'label': 1})
    return df

# switches human labels to 0 (negative) and bot labels to 1 (positive)
def fix_positive_condition(df):
    df = df.replace({'label': 1}, {'label': 3})
    df = df.replace({'label': 0}, {'label': 1})
    df = df.replace({'label': 3}, {'label': 0})
    return df


def logistic_regression(path):
    master_df = pd.read_csv(path)

    x1 = master_df.drop(['label'], axis=1).values
    y1 = master_df['label'].values

    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(x1, y1, test_size=0.30, random_state=100)

    X_train_scaled = preprocessing.scale(X_train)
    X_test_scaled = preprocessing.scale(X_test)
    model = LogisticRegression()
    model.fit(X_train_scaled, Y_train)
    result = model.score(X_test_scaled, Y_test)
    # print("Accuracy: %.2f%%" % (result * 100.0))

    return result


def remove_outliers(zscore, pathA, pathB):
    master_df = pd.read_csv(pathA)
    master_df = fix_positive_condition(master_df)
    master_df = orgs_to_bots(master_df)
    columns = ["astroturf", "fake follower", "financial", "other", "overall", "self-declared", "spammer", "label"]
    master_df = master_df[columns]
    human_df = master_df[master_df['label'] == 0]
    bot_df = master_df[master_df['label'] == 1]

    columns = ["astroturf", "fake follower", "financial", "other", "overall", "self-declared", "spammer"]
    human_df = human_df[np.abs(stats.zscore(human_df[columns]) < zscore).all(axis=1)]
    bot_df = bot_df[np.abs(stats.zscore(bot_df[columns]) < zscore).all(axis=1)]

    master_df = pd.concat([human_df, bot_df])

    master_df.to_csv(pathB, index=False)


def confusion_matrix(path, lang):
    master_df = pd.read_csv(path)

    x1 = master_df.drop(['label'], axis=1).values
    y1 = master_df['label'].values

    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(x1, y1, test_size=0.30, random_state=100)

    X_train_scaled = preprocessing.scale(X_train)
    X_test_scaled = preprocessing.scale(X_test)
    model = LogisticRegression()
    model.fit(X_train_scaled, Y_train)

    title_options = [("Confusion Matrix Without Normalization", None), ("Normalization: True", "true"),
                     ("Normalization: Prediction", "pred"), ("Normalization: All", "all")]
    for title, normalize in title_options:
        disp = ConfusionMatrixDisplay.from_estimator(model, X_test_scaled, Y_test, display_labels=["bot", "human"],
                                              cmap=plt.cm.Blues, normalize=normalize)
        file_path = r'C:\Users\Esau\PycharmProjects\PersonalProject\{}_{}'.format(normalize, lang)
        disp.ax_.set_title(title)
        plt.savefig(file_path)

# gets zscore graph
zscore = 3
zscore_list = []
results_list = []
while zscore >= 0:
    zscore_list.append(zscore)
    remove_outliers(zscore, universal, universal_no_outliers)
    result = logistic_regression(universal_no_outliers)
    result *= 100
    results_list.append(float("{:.2f}".format(result)))
    #print(f"{zscore} processed")
    zscore -= 0.1

plt.plot(zscore_list, results_list)
plt.xlabel('zscore')
plt.ylabel('accuracy (%)')

remove_outliers(0.701, english, english_no_outliers)
logistic_regression(english_no_outliers)
confusion_matrix(english_no_outliers, "eng")

remove_outliers(1.298, universal, universal_no_outliers)
logistic_regression(universal_no_outliers)
confusion_matrix(universal_no_outliers, "univ")

plt.show()