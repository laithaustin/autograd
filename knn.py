import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

class knn_classifier():
    def __init__(self, k):
        self.k = k
        self.x = None
        self.y = None
            
    def fit(self, x, y):
        """
        Initializes our data and labels for further prediciton
        """
        # TODO: implement using kdtrees
        self.x = x
        self.y = y
    
    def _knn(self, point):
        """
        Computes euclidean distances and returns best prediction
        """
        distances = np.linalg.norm(self.x - point, axis = 1)
        # append labels to distances and sort by top k labels
        app = np.array([distances, self.y]).T
        app = np.array(app[app[:,0].argsort()][:self.k])
        # returns the largest bin of labels
        return np.argmax(np.bincount(app[:,1].astype(int)))

    def predict(self, test):
        """
        Runs inference on test set by computing Euclidean distances
        """
        preds = [self._knn(p) for p in test]
        return preds

        
# some sample tests
iris = datasets.load_iris()
x = iris.data
y = iris.target

# split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

# train model
model = knn_classifier(5)
model.fit(x_train, y_train)

# compute predictions and compute accuracy
preds = model.predict(x_test)
accs = np.sum(preds == y_test) / len(y_test)
print("Accuracy: ", accs)
