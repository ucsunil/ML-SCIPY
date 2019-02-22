import matplotlib as mplt
mplt.use('TkAgg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
from mlalgorithms.adalinegd import AdalineGD

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
y = df.iloc[0:100, 4].values
# print(y)

y = np.where(y == 'Iris-setosa', -1, 1)
# print(y)

X = df.iloc[0:100, [0,2]].values

fig, ax = plot.subplots(nrows=1, ncols=2, figsize=(8, 4))
ada1 = AdalineGD(n_iter=10, eta=0.01).fit(X, y)

ax[0].plot(range(1, len(ada1.cost_)+1), np.log10(ada1.cost_), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-error)')
ax[0].set_title('Adaline - Learning rate 0.01')

ada2 = AdalineGD(n_iter=10, eta=0.0001).fit(X, y)
ax[1].plot(range(1, len(ada2.cost_)+1), ada2.cost_, marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Sum-squared-error')
ax[1].set_title('Adaline - Learning rate 0.0001')

plot.show()

X_std = np.copy(X)
X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()

from processing.perceptrontest import plot_decision_regions

adal = AdalineGD(n_iter=15, eta=0.01)
adal.fit(X_std, y)
plot_decision_regions(X_std, y, classifier=adal)
plot.title('Adaline - Gradient Descent')
plot.xlabel('sepal length [standardized]')
plot.ylabel('petal length [standardized]')
plot.legend(loc='upper left')
plot.show()

plot.plot(range(1, len(adal.cost_)+1), adal.cost_, marker='o')
plot.xlabel('Epochs')
plot.ylabel('Sum-squared-error')
plot.show()