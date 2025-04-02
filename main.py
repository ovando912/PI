import sys

sys.path.append("/home/lucas/Documents/Proyecto_Integrador/PI")
from functions import *
path = "/home/lucas/Documents/Proyecto_Integrador/PI/segundo_semestre/3-19-25"

if not os.path.exists(path):
    os.makedirs(path)
source_path = (
    "/home/lucas/Documents/Proyecto_Integrador/PI/segundo_semestre/sources/source1/"
)
os.chdir(path)

# Inputs:

geometria = [True, 15, 15, 100, 3, 3]
z0 = 30
fuente_original = ["monoenergetica", "colimada"]

N_original = int(1e6)

columns_order = ["ln(E0/E)", "x", "y", "mu", "phi"]
micro_bins = [500] * len(columns_order)
# micro_bins = [50000, 40000, 35000,35000,30000]
macro_bins = [10, 8, 8, 7]
type = "equal_bins"
used_defined_edges = [
    [2.995732273553991 + 1e-9],
    # None,
    [-1.5, 1.5],
    [-1.5, 1.5],
    [1 - 1e-9],
    # None,
    None,
]

# N_file = int(
#     5e6
# )  # toma esta cantidad del archivo fuente, aunque el archivo fuente puede ser mas grande


print("Extracting data from source file")
df, factor_normalizacion, cantidad_registrada = df_from_source_file(
    source_path + "surface_source.h5", N_original
)

plot_correlated_variables(df, columns_order, filename="original.png")

config = {
    "geometria": geometria,
    "z0": z0,
    "N_original": N_original,
    "fuente_original": fuente_original,
    "columns_order": columns_order,
    "micro_bins": micro_bins,
    "macro_bins": macro_bins,
    "binning_type": type,
    "used_defined_edges": used_defined_edges,
    "factor_normalizacion": factor_normalizacion,
}

# Imprime la cantidad de particulas registrada:
print(f"Particulas registradas: {cantidad_registrada:.1e}")
# Imprime la cantidad de particulas registradas con mu = 1 en porcentaje:
print(
    f"Particulas registradas con mu = 1: {df.loc[df['mu'] == 1].shape[0] / cantidad_registrada * 100:.2f}%"
)

print("Creating source")
nodo = calculate_cumul_micro_macro_nodo(
    df,
    columns_order,
    micro_bins,
    macro_bins,
    binning_type=type,
    user_defined_macro_edges=used_defined_edges,
)
del df
print("Saving source")

nodo.save_to_xml("source.xml", config=config)