import argparse
import matplotlib.pyplot as plt
import graphviz
import pandas as pd
import numpy as np
from sklearn import preprocessing, tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# for displaying decision tree classifier graph using graphviz library
def pretty_graph(dtc, attributes, le):
    """Generate a dot file and render it to vizualize the decision tree

    Parameters
    ----------
    dtc: DecisionTreeClassifier
        The deicision tree classifier object

    attributes: list
        attributes

    le: labelEncoder
        label encoder
    """
    dot_data = tree.export_graphviz(
        dtc,
        out_file=None,
        feature_names=attributes,
        class_names=le.classes_,
        filled=True,
        rounded=True,
    )
    graph = graphviz.Source(dot_data)
    graph.render("mytree1")
    graph.view()


def main():

    # Get dataset file
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=argparse.FileType("r"),
        required=True,
        help="Dataset to train as .csv file",
    )
    parser.add_argument(
        "--predict", type=argparse.FileType("r"), help="File with data to predict"
    )
    parser.add_argument("--display", default=False, action="store_true")
    arg = parser.parse_args()
    dataset_file = arg.dataset
    predict_file = arg.predict
    display_graph = arg.display

    # Load training dataset
    print("Loading dataset...")
    df = pd.read_csv(dataset_file, header=0)
    for col in df.columns:
        try:
            df[col] = df[col].str.strip()
        except:
            pass

    attributes = list(df.columns.values)[:-1]

    # Divide columns into features column and label column
    end = df.values.shape[1] - 1
    X = df.values[:, 0:end]
    Y = df.values[:, end]

    # Preprocess data to transform data from str to number classes
    X_encode = np.empty_like(X)
    encoding = list()
    print("Preprocessing dataset for training...")
    le = preprocessing.LabelEncoder()
    for i in range(len(X[0])):
        X_encode[:, i] = le.fit_transform(X[:, i])
        d = dict()
        for key, value in zip(X[:, i], X_encode[:, i]):
            if key not in d:
                d[key] = value
        encoding.append(d)
    y = le.fit_transform(Y)

    # Split the data set into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_encode, Y, test_size=0.3, random_state=42
    )

    # Create a classifier object using sklearn - tree module
    dtc = tree.DecisionTreeClassifier(
        criterion="entropy",
        splitter="best",
        max_depth=6,
        max_features=20,
        min_samples_split=5,
        max_leaf_nodes=6,
        random_state=100,
    )

    # Train the classifier
    print("Training the dataset...")
    dtc.fit(X_train, y_train)

    # Test
    y_pred = dtc.predict(X_test)

    # Obtain accuracy of the model
    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
    print("Training is finalized:")
    print("Model Accuracy: ", accuracy * 100, "%")

    # Visualize graph
    tree.plot_tree(dtc)
    if display_graph:
        plt.show()
    plt.savefig("dtc.png")

    try:
        pretty_graph(dtc, attributes, le)
    except:
        print("graphviz not installed!")

    # Load data to predict
    p_df = pd.read_csv(predict_file, header=None)
    print(p_df)

    # Encode prediction data based on labels created
    print("Processing prediction...")
    encoded_pred = np.empty_like(p_df)
    row, col = p_df.values.shape

    for i in range(row):
        for j in range(col):
            v = p_df.values[i, j]

            if isinstance(v, str):
                v = v.strip()

            encoded_pred[i][j] = encoding[j][v]
    print(encoded_pred)

    # Predict
    prediction = dtc.predict(encoded_pred)
    print("Predicted output - ", list(df.columns.values)[-1], ": ", prediction)


if __name__ == "__main__":
    main()
