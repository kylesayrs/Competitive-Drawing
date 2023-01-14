import pickle

from competitive_drawing.train.utils import Classifier

if __name__ == "__main__":
    file_path = "./model.pkl"

    with open(file_path, "wb") as file:
        pickle.dump(Classifier, file)
