import pickle
from model import Classifier

cls = Classifier(epochs=64, batch_size=32, metrics=True, plot_model_diagram=True, summary=True)

with open("data.pkl","rb") as pickle_in:
  data = pickle.load(pickle_in)
pickle_in.close()

cls.train(data)