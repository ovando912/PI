# %load_ext autoreload
# %autoreload 2
import sys

sys.path.append("../")
from functions import *

# %matplotlib widget

# Data to test
columns_order = [
    ["ln(E0/E)", "x", "y", "mu", "phi"],
]
micro_bins = [
    [200] * len(columns_order[0]),
    [250] * len(columns_order[0]),
    [300] * len(columns_order[0]),
]
macro_bins = [[15, 10, 8, 6, 5], [12, 12, 12, 12, 12]]
N_max = [1e7]
type = ["equal_area"]

columns_order, micro_bins, macro_bins, N, type = barrido_combinations(
    columns_order, micro_bins, macro_bins, N_max, type
)
N_max = [n.sum() for n in N]

save_information(type, N_max, columns_order, micro_bins, macro_bins)

SurfaceSourceFile = kds.SurfaceSourceFile(
    "../surface_source.mcpl", domain={"w": [0, 1]}
)
df = SurfaceSourceFile.get_pandas_dataframe()
df = df[["x", "y", "ln(E0/E)", "mu", "phi", "wgt"]]
del SurfaceSourceFile

# Run test
barrido(
    columns_order,
    micro_bins,
    macro_bins,
    N,
    type,
    df,
    save=True,
)

# Graficar comparacion
comparacion_barrido()