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


def format_df_to_MCPL(
    df: pd.DataFrame,
    z: float,
    time: float = 0,
    wgt: float = 1,
    delayed_group: float = 0,
    particle: int = 2112,
    E_0: float = 20,
) -> pd.DataFrame:
    
    # Vectorized calculations
    df["theta"] = np.arccos(df["mu"])
    df["E"] = E_0 * np.exp(-df["ln(E0/E)"])  # MeV
    df["u_x"] = np.cos(df["phi"]) * np.sin(df["theta"])
    df["u_y"] = np.sin(df["phi"]) * np.sin(df["theta"])
    df["u_z"] = np.cos(df["theta"])

    # Construct the result DataFrame with all columns in one step
    df_result = pd.DataFrame({
        "id": df.index,
        "type": particle,
        "E": df["E"],
        "x": df["x"],
        "y": df["y"],
        "z": z,
        "u": df["u_x"],
        "v": df["u_y"],
        "w": df["u_z"],
        "t": time,
        "wgt": wgt,
        "px": 0,
        "py": 0,
        "pz": 0,
        "userflags": delayed_group,
    })

    # Ensure integer columns are correctly typed
    df_result["id"] = df_result["id"].astype(int)
    df_result["type"] = df_result["type"].astype(int)
    df_result["userflags"] = df_result["userflags"].astype(int)

    return df_result

df_formatted = format_df_to_MCPL(df_sampled, z=5)
print(df_formatted.head())

kds.create_source_file(df_formatted, "creado.h5")