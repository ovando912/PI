# %load_ext autoreload
# %autoreload 2
import sys

sys.path.append("../")
from functions import *

# %matplotlib widget

# Data to test
columns_order = [
    ["ln(E0/E)", "x", "y", "mu", "phi"],
    ["x", "y", "ln(E0/E)", "mu", "phi"],
]
micro_bins = [[200] * len(columns_order[0]),
              [300]*len(columns_order[0]),              
              ]
macro_bins = [[15, 10, 8, 6, 5],
              [10,10,10,10,10],
              ]
N_max = [1e6]
type = ["equal_area"]

columns_order, micro_bins, macro_bins, N, type = barrido_combinations(
    columns_order, micro_bins, macro_bins, N_max, type
)
N_max = [n.sum() for n in N]

save_information(type, N_max, columns_order, micro_bins, macro_bins)

SurfaceSourceFile = kds.SurfaceSourceFile(
    "/home/lucas/Proyecto_Integrador/Proyecto_Integrador/track_files/surface_source_1.h5",
    domain={"w": [0, 2]},
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
    bins_for_comparation=500,
    value_to_replace_0=None,
)

get_min_max(len(columns_order),len(columns_order[0]))

comparacion_barrido(len(columns_order))

combine_images(len(columns_order),f'{int(N_max[0])}.png')
