
import numpy as np
import plotly.graph_objects as go
import sklearn as skl
from sklearn.decomposition import PCA

np.set_printoptions(precision=3, suppress=True)

colors = [
    np.array((31,119,180), dtype=np.float64),
    np.array((190, 68, 255), dtype=np.float64),
    np.array((44, 160, 44), dtype=np.float64),
    np.array((214, 39, 40), dtype=np.float64),
]

for c in colors:
    c = c/255
    # round to 2 decimal places
    c = np.round(c, 2)
    print(f"rgba({c[0]}, {c[1]}, {c[2]}, 0.2)")



rgba(0.12, 0.47, 0.71, 0.2)
rgba(0.75, 0.27, 1.0, 0.2)
rgba(0.17, 0.63, 0.17, 0.2)
rgba(0.84, 0.15, 0.16, 0.2)