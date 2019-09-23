import os
import pickle
import pprint
import sys

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42

config = dict(
    IN_FOLDER=".",
    FEATURES_FILE_NAME="mnist_features.pkl",
    LABELS_FILE_NAME="mnist_y.pkl",
    OUT_FOLDER=".",
    MODEL_FILE_NAME="model.pkl",
    MODEL_ID="",
)


if __name__ == "__main__":
    pp = pprint.PrettyPrinter(indent=4)
    config.update({k: os.environ[k] for k in config.keys() if k in os.environ.keys()})
    print(f"configuration: {config}")
    print("Env:")
    pp.pprint(dict(os.environ))
    print("folder structure")
    pp.pprint(list(os.walk('.')))

    features_path = os.path.join(config['IN_FOLDER'], config['FEATURES_FILE_NAME'])
    labels_path = os.path.join(config['IN_FOLDER'], config['LABELS_FILE_NAME'])
    model_path = os.path.join(config['OUT_FOLDER'], config['MODEL_FILE_NAME'])

    print("Loading features")
    with open(features_path, 'rb') as infeatures:
        features = pickle.load(infeatures)

    print("Loading labels")
    with open(labels_path, 'rb') as inlabels:
        labels = pickle.load(inlabels)

    # Split Dataset
    print(f"Features shape: {features.shape}; Labels shape: {labels.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.33, random_state=RANDOM_STATE
    )
    print(f"Train shapes: {X_train.shape} {y_train.shape}")
    print(f"Test shapes: {X_test.shape} {y_test.shape}")

    # Train
    print("Training...")
    rf = RandomForestClassifier()
    rf = rf.fit(X_train, y_train)
    
    # Evaluate
    print("Evaluation...")
    predictions = rf.predict(X_test)
    
    print("Confusion Matrix")
    print(f"{confusion_matrix(y_test, predictions)}")
    print("")
    print("Classification Report")
    print(f"{classification_report(y_test, predictions)}")
    print("")

    print(f'Saving model to "{model_path}"')
    # Save the model
    with open(model_path, 'wb') as model_file:
        pickle.dump(rf, model_file)

    sys.exit(0)
