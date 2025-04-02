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
    [75] * len(columns_order[0]),
    [100] * len(columns_order[0]),
    [125] * len(columns_order[0]),
    [150] * len(columns_order[0]),
    [175] * len(columns_order[0]),
    [200] * len(columns_order[0]),
    [225] * len(columns_order[0]),
    [250] * len(columns_order[0]),
    [275] * len(columns_order[0]),
    [300] * len(columns_order[0]),
    [325] * len(columns_order[0]),
    [350] * len(columns_order[0]),
    [375] * len(columns_order[0]),
    [400] * len(columns_order[0]),
    [425] * len(columns_order[0]),
]
macro_bins = [[15, 10, 8, 6, 5]]
N_max = [1e6]
type = ["equal_area"]

columns_order, micro_bins, macro_bins, N, type = barrido_combinations(
    columns_order, micro_bins, macro_bins, N_max, type
)
N_max = [n.sum() for n in N]

save_information(type, N_max, columns_order, micro_bins, macro_bins)

SurfaceSourceFile = kds.SurfaceSourceFile(
    "/home/lucas/Proyecto_Integrador/track_files/surface_source.mcpl", domain={"w": [0, 1]}
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