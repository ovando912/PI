# %load_ext autoreload
# %autoreload 2
import sys

sys.path.append("../")
from functions import *

# %matplotlib widget

# Data to test
columns_order = [
    ["x", "y", "ln(E0/E)", "mu", "phi"],
    ["ln(E0/E)", "x", "y", "mu", "phi"],
    ["mu", "ln(E0/E)", "x", "y", "phi"],
    ["phi", "mu", "ln(E0/E)", "x", "y"],
    ["mu", "x", "y", "ln(E0/E)", "phi"],
]
micro_bins = [
    [100] * len(columns_order[0]),
    [150] * len(columns_order[0]),
    [200] * len(columns_order[0]),
]
macro_bins = [[15, 10, 8, 6, 5], 
              [10, 10, 10, 10, 10], 
              [5, 6, 8, 10, 15]]
N_max = [1e7]
type = ["equal_area", "equal_bins"]

columns_order, micro_bins, macro_bins, N, type = barrido_combinations(
    columns_order, micro_bins, macro_bins, N_max, type
)
N_max = [n.sum() for n in N]

save_information(type, N_max, columns_order, micro_bins, macro_bins)

SurfaceSourceFile = kds.SurfaceSourceFile(
    "../track_files/surface_source.mcpl", domain={"w": [0, 2]}
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

# comparacion_barrido()
