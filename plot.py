import sys
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

cifar10_layers = ["10", "11", "12"]
cifar10_labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
cm = plt.get_cmap('viridis')

NUM_COLORS = 10
for method in ["pca", "tsne"]:
    for layer in cifar10_layers:
        for dimension in [2, 3]:
            data = pickle.load(open(f"cifar_{method}_layer{layer}_{dimension}d.pickle", "rb"))
            fig = plt.figure()
            if dimension == 2:
                ax = fig.add_subplot()
            if dimension == 3:
                ax = fig.add_subplot(projection='3d')
                ax.view_init(elev=10, azim=-127)

            ax.set_prop_cycle(color=[cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])
            for l in cifar10_labels:
                d = data.loc[data["label"]==l]
                if dimension == 2:
                    ax.scatter(d["x"], d["y"])
                if dimension == 3:
                    ax.scatter(d["x"], d["y"], d["z"])
            plt.savefig(f"cifar_{method}_{dimension}d_layer{layer}.pdf")
            # plt.show()

