import sys

sys.path.append("/home/lucas/Proyecto_Integrador/Proyecto_Integrador")
from functions import *


# Data to test
columns_order = ["x", "y", "ln(E0/E)", "mu", "phi"]
micro_bins = [300] * len(columns_order)
macro_bins = [10, 8, 6, 4]
N_max = 3147511
type = "equal_area"

filename = "/home/lucas/Proyecto_Integrador/Paralelepipedo/surface_source.h5"


SurfaceSourceFile = kds.SurfaceSourceFile(
    filename,
    domain={"w": [0, 2]},
)
df = SurfaceSourceFile.get_pandas_dataframe()
df = df[["x", "y", "ln(E0/E)", "mu", "phi", "wgt", "E"]]
del SurfaceSourceFile

print(len(df))
# plot_correlated_variables(df, columns_order)

cumul, micro, macro = calculate_cumul_micro_macro(
    df, columns_order, micro_bins, macro_bins, type=type
)
df_sampled = sample(cumul, micro, macro, columns_order, N_max)

# plot_correlated_variables(df_sampled, columns_order)

MCPLColumns = [
    "id",
    "type",
    "E",
    "x",
    "y",
    "z",
    "u",
    "v",
    "w",
    "t",
    "wgt",
    "px",
    "py",
    "pz",
    "userflags",
]




df_formatted = format_df_to_MCPL(df_sampled, z=5)
print(df_formatted.head())

kds.create_source_file(df_formatted, "creado.h5")