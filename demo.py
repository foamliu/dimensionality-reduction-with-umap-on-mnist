import umap

from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import seaborn as sns
import mplcursors

sns.set(context="paper", style="white")

mnist = fetch_openml("mnist_784", version=1)

reducer = umap.UMAP(random_state=42)
embedding = reducer.fit_transform(mnist.data)

fig, ax = plt.subplots(figsize=(12, 10))
color = mnist.target.astype(int)
plt.scatter(embedding[:, 0], embedding[:, 1], c=color, cmap="Spectral", s=0.1)
plt.setp(ax, xticks=[], yticks=[])
plt.title("MNIST data embedded into two dimensions by UMAP", fontsize=18)

mplcursors.cursor(hover=True)
plt.show()
#
# import matplotlib.pyplot as plt
# import numpy as np
# import mplcursors
# np.random.seed(42)
#
# fig, ax = plt.subplots()
# ax.scatter(*np.random.random((2, 26)))
# ax.set_title("Mouse over a point")
#
# mplcursors.cursor(hover=True)
#
# plt.show()