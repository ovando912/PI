import sys

sys.path.append("/home/lucas/Documents/Proyecto_Integrador/PI")
from functions import *

path = "/home/lucas/Documents/Proyecto_Integrador/PI/segundo_semestre/12-2-25"
os.chdir(path)

fuente_original = ["monoenergetica", "colimada"]
geometria = [True, 15, 15, 100, 3, 3]
z0 = 30
N_original = 2.5e6*20*9

# files_to_remove = ['geometry.xml', 'materials.xml', 'settings.xml', 'tallies.xml', 'original.png','statepoint_original.h5','summary.h5','surface_source.h5','tallies.out','sintetico.png','statepoint_sintetico.h5']

run_simulation(fuente_original, geometria, z0, int(N_original))

SurfaceSourceFile = kds.SurfaceSourceFile("surface_source.h5", domain={"w": [0, 2]})
df = SurfaceSourceFile.get_pandas_dataframe()
df = df[["x", "y", "ln(E0/E)", "mu", "phi", "wgt"]]
del SurfaceSourceFile

factor_normalizacion = df["wgt"].sum() / N_original
cantidad_registrada = len(df)

columns_order = ["ln(E0/E)", "x", "y", "mu", "phi"]
micro_bins = [2000] * len(columns_order)
# micro_bins = [50000, 40000, 35000,35000,30000]
macro_bins = [9, 6, 6, 4]
N_sintetico = 2e7
type = "equal_bins"


bins_for_comparation: int = 150
value_to_replace_0: float = 1e-6

# Calculate the original edges and original counts
edges_1d, edges_2d_1, edges_2d_2 = get_bin_edges(
    df, columns_order, bin_size=bins_for_comparation
)

counts_1d_original, counts_2d_original = get_counts(
    df, columns_order, edges_1d, edges_2d_1, edges_2d_2
)


# normalize counts_container and take aways the zeros
counts_1d_original, counts_2d_original = normalize_counts(
    counts_1d_original,
    counts_2d_original,
    edges_1d,
    edges_2d_1,
    edges_2d_2,
    value_to_replace_0=value_to_replace_0,
)

# Save the original histograms
plot_correlated_variables_counts(
    counts_1d=counts_1d_original,
    counts_2d=counts_2d_original,
    edges_1d=edges_1d,
    edges_2d_1=edges_2d_1,
    edges_2d_2=edges_2d_2,
    columns_order=columns_order,
    filename=f"original.png",
)

del counts_1d_original, counts_2d_original

used_defined_edges = [
    [2.995732273553991 + 1e-9],
    # None,
    [-1.5, 1.5],
    [-1.5, 1.5],
    [1 - 1e-9],
    # None,
    None,
]
print("flag 1")
cumul, micro, macro = calculate_cumul_micro_macro(
    df,
    columns_order,
    micro_bins,
    macro_bins,
    binning_type=type,
    user_defined_macro_edges=used_defined_edges,
)
del df
print("flag 2")
df_sampled = sample(cumul, micro, macro, columns_order, int(N_sintetico))
del cumul, micro, macro
print("flag 3")
(
    counts_1d_sintetico,
    counts_2d_sintetico,
) = get_counts(
    df_sampled,
    columns_order,
    edges_1d,
    edges_2d_1,
    edges_2d_2,
)

counts_1d_sintetico_normalized, counts_2d_sintetico_normalized = normalize_counts(
    counts_1d_sintetico,
    counts_2d_sintetico,
    edges_1d,
    edges_2d_1,
    edges_2d_2,
    value_to_replace_0=value_to_replace_0,
)

plot_correlated_variables_counts(
    counts_1d=counts_1d_sintetico_normalized,
    counts_2d=counts_2d_sintetico_normalized,
    edges_1d=edges_1d,
    edges_2d_1=edges_2d_1,
    edges_2d_2=edges_2d_2,
    columns_order=columns_order,
    filename=f"sintetico.png",
)
del edges_1d, edges_2d_1, edges_2d_2
del (
    counts_1d_sintetico,
    counts_2d_sintetico,
    counts_1d_sintetico_normalized,
    counts_2d_sintetico_normalized,
)

df_formatted = format_df_to_MCPL(df_sampled, z=z0)
del df_sampled


kds.create_source_file(df_formatted, "sintetico.h5")
del df_formatted

fuente_sintetica = ["sintetico.h5"]

run_simulation(fuente_sintetica, geometria, z0, int(N_sintetico))

# Plot the results

# Load the statepoint file 1
sp = openmc.StatePoint("statepoint_original.h5")
tally_total = sp.get_tally(name="flux_total")
df_original_total = tally_total.get_pandas_dataframe()
df_original_total.columns = [
    "x",
    "y",
    "z",
    "nuclide",
    "score",
    "mean",
    "std.dev.",
]
tally_total = sp.get_tally(name="espectro_total")
df_original_espectro_total = tally_total.get_pandas_dataframe()
df_original_espectro_total.columns = [
    "x",
    "y",
    "z",
    "E_min",
    "E_max",
    "nuclide",
    "score",
    "mean",
    "std.dev.",
]
if geometria[0]:
    tally_vacio = sp.get_tally(name="flux_vacio")
    df_original_vacio = tally_vacio.get_pandas_dataframe()
    df_original_vacio.columns = ["x", "y", "z", "nuclide", "score", "mean", "std.dev."]
    tally_vacio = sp.get_tally(name="espectro_vacio")
    df_original_espectro_vacio = tally_vacio.get_pandas_dataframe()
    df_original_espectro_vacio.columns = [
        "x",
        "y",
        "z",
        "E_min",
        "E_max",
        "nuclide",
        "score",
        "mean",
        "std.dev.",
    ]

# Load the statepoint file 2
sp = openmc.StatePoint("statepoint_sintetico.h5")
tally_total = sp.get_tally(name="flux_total")
df_sintetico_total = tally_total.get_pandas_dataframe()
df_sintetico_total.columns = ["x", "y", "z", "nuclide", "score", "mean", "std.dev."]
tally_total = sp.get_tally(name="espectro_total")
df_sintetico_espectro_total = tally_total.get_pandas_dataframe()
df_sintetico_espectro_total.columns = [
    "x",
    "y",
    "z",
    "E_min",
    "E_max",
    "nuclide",
    "score",
    "mean",
    "std.dev.",
]
if geometria[0]:
    tally_vacio = sp.get_tally(name="flux_vacio")
    df_sintetico_vacio = tally_vacio.get_pandas_dataframe()
    df_sintetico_vacio.columns = ["x", "y", "z", "nuclide", "score", "mean", "std.dev."]
    tally_vacio = sp.get_tally(name="espectro_vacio")
    df_sintetico_espectro_vacio = tally_vacio.get_pandas_dataframe()
    df_sintetico_espectro_vacio.columns = [
        "x",
        "y",
        "z",
        "E_min",
        "E_max",
        "nuclide",
        "score",
        "mean",
        "std.dev.",
    ]

factor = len(df_original_total) / geometria[3]
z_min = int(z0 * factor)
z_max = int(
    max(
        df_original_total.loc[df_original_total["mean"] != 0, "z"].max(),
        df_sintetico_total.loc[df_sintetico_total["mean"] != 0, "z"].max(),
    )
    * 1.1
)

# Crear los gráficos
if geometria[0]:
    fig, axs = plt.subplots(3, 4, figsize=(24, 10))
    plt.subplots_adjust(
        left=0.1, right=0.9, top=0.75, bottom=0.1, wspace=0.4, hspace=0.4
    )

else:
    fig, axs = plt.subplots(1, 4, figsize=(24, 7))
    plt.subplots_adjust(top=0.75, wspace=0.4)

axs = np.atleast_2d(axs)  # Convierte axs en una matriz 2D siempre

# Primer gráfico: Flux vs. z
axs[0, 0].plot(
    df_original_total["z"][z_min:z_max] / factor,
    df_original_total["mean"][z_min:z_max],
    label="Original",
)
axs[0, 0].plot(
    df_sintetico_total["z"][z_min:z_max] / factor,
    df_sintetico_total["mean"][z_min:z_max] * factor_normalizacion,
    label="Sintético",
)
axs[0, 0].set_xlabel("z [cm]")
axs[0, 0].set_ylabel("Flux")
axs[0, 0].set_yscale("log")
axs[0, 0].minorticks_on()  # Habilitar marcas menores
axs[0, 0].grid(which="both", linestyle="--", linewidth=0.5)
axs[0, 0].set_title("Flux vs. z")
axs[0, 0].legend()

# Segundo gráfico: Error relativo
y_data = (
    (
        df_original_total["mean"][z_min:]
        - df_sintetico_total["mean"][z_min:] * factor_normalizacion
    )
    / df_original_total["mean"][z_min:]
    * 100
)
axs[0, 1].plot(
    df_original_total["z"][z_min:] / factor,
    y_data,
)
axs[0, 1].set_xlabel("z [cm]")
axs[0, 1].set_ylabel("Error relativo [%]")
axs[0, 1].minorticks_on()  # Habilitar marcas menores
axs[0, 1].grid(which="both", linestyle="--", linewidth=0.5)
axs[0, 1].set_title("Error relativo vs. z")
axs[0, 1].set_ylim(max(y_data.min(), -100), min(y_data.max(), 100))

if geometria[0]:
    # Tercer gráfico: Flux agua vs. z
    axs[1, 0].plot(
        df_original_vacio["z"][z_min:z_max] / factor,
        df_original_total["mean"][z_min:z_max] - df_original_vacio["mean"][z_min:z_max],
        label="Original",
    )
    axs[1, 0].plot(
        df_sintetico_vacio["z"][z_min:z_max] / factor,
        df_sintetico_total["mean"][z_min:z_max] * factor_normalizacion
        - df_sintetico_vacio["mean"][z_min:z_max] * factor_normalizacion,
        label="Sintético",
    )
    axs[1, 0].set_xlabel("z [cm]")
    axs[1, 0].set_ylabel("Flux agua")
    axs[1, 0].set_yscale("log")
    axs[1, 0].minorticks_on()  # Habilitar marcas menores
    axs[1, 0].grid(which="both", linestyle="--", linewidth=0.5)
    axs[1, 0].set_title("Flux agua vs. z")
    axs[1, 0].legend()

    # Cuarto gráfico: Error relativo agua
    y_data = (
        (
            df_original_total["mean"][z_min:]
            - df_original_vacio["mean"][z_min:]
            - df_sintetico_total["mean"][z_min:] * factor_normalizacion
            + df_sintetico_vacio["mean"][z_min:] * factor_normalizacion
        )
        / (df_original_total["mean"][z_min:] - df_original_vacio["mean"][z_min:])
        * 100
    )
    axs[1, 1].plot(
        df_original_vacio["z"][z_min:] / factor,
        y_data,
    )
    axs[1, 1].set_xlabel("z [cm]")
    axs[1, 1].set_ylabel("Error relativo [%]")
    axs[1, 1].minorticks_on()  # Habilitar marcas menores
    axs[1, 1].grid(which="both", linestyle="--", linewidth=0.5)
    axs[1, 1].set_title("Error relativo agua")
    axs[1, 1].set_ylim(max(y_data.min(), -100), min(y_data.max(), 100))

    # Quinto gráfico: flux vacio vs. z
    axs[2, 0].plot(
        df_original_vacio["z"][z_min:z_max] / factor,
        df_original_vacio["mean"][z_min:z_max],
        label="Original",
    )
    axs[2, 0].plot(
        df_sintetico_vacio["z"][z_min:z_max] / factor,
        df_sintetico_vacio["mean"][z_min:z_max] * factor_normalizacion,
        label="Sintético",
    )
    axs[2, 0].set_xlabel("z [cm]")
    axs[2, 0].set_ylabel("Flux vacio")
    axs[2, 0].set_yscale("log")
    axs[2, 0].minorticks_on()  # Habilitar marcas menores
    axs[2, 0].grid(which="both", linestyle="--", linewidth=0.5)
    axs[2, 0].set_title("Flux vacio vs. z")
    axs[2, 0].legend()

    # Sexto gráfico: Error relativo vacio
    y_data = (
        (
            df_original_vacio["mean"][z_min:]
            - df_sintetico_vacio["mean"][z_min:] * factor_normalizacion
        )
        / df_original_vacio["mean"][z_min:]
        * 100
    )
    axs[2, 1].plot(
        df_original_vacio["z"][z_min:] / factor,
        y_data,
    )
    axs[2, 1].set_xlabel("z [cm]")
    axs[2, 1].set_ylabel("Error relativo [%]")
    axs[2, 1].minorticks_on()  # Habilitar marcas menores
    axs[2, 1].grid(which="both", linestyle="--", linewidth=0.5)
    axs[2, 1].set_title("Error relativo vacio")
    axs[2, 1].set_ylim(max(y_data.min(), -100), min(y_data.max(), 100))

# Séptimo gráfico: Espectro total
axs[0,2].plot(df_original_espectro_total["E_min"] + 0.5 * (df_original_espectro_total["E_max"] - df_original_espectro_total["E_min"]), df_original_espectro_total["mean"], label="Original")
axs[0,2].plot(df_sintetico_espectro_total["E_min"] + 0.5 * (df_sintetico_espectro_total["E_max"] - df_sintetico_espectro_total["E_min"]), df_sintetico_espectro_total["mean"] * factor_normalizacion, label="Sintético")
axs[0,2].set_xlabel("Energía [eV]")
axs[0,2].set_ylabel("Flujo")
axs[0,2].set_yscale("log")
axs[0,2].set_xscale("log")
axs[0,2].minorticks_on()  # Habilitar marcas menores
axs[0,2].grid(which="both", linestyle="--", linewidth=0.5)
axs[0,2].set_title("Espectro total")
axs[0,2].legend()

if geometria[0]:
    # Octavo gráfico: Espectro agua
    axs[1,2].plot(df_original_espectro_vacio["E_min"] + 0.5 * (df_original_espectro_vacio["E_max"] - df_original_espectro_vacio["E_min"]), df_original_espectro_total["mean"] - df_original_espectro_vacio["mean"], label="Original")
    axs[1,2].plot(df_sintetico_espectro_vacio["E_min"] + 0.5 * (df_sintetico_espectro_vacio["E_max"] - df_sintetico_espectro_vacio["E_min"]), df_sintetico_espectro_total["mean"] * factor_normalizacion - df_sintetico_espectro_vacio["mean"] * factor_normalizacion, label="Sintético")
    axs[1,2].set_xlabel("Energía [eV]")
    axs[1,2].set_ylabel("Flujo")
    axs[1,2].set_yscale("log")
    axs[1,2].set_xscale("log")
    axs[1,2].minorticks_on()  # Habilitar marcas menores
    axs[1,2].grid(which="both", linestyle="--", linewidth=0.5)
    axs[1,2].set_title("Espectro agua")
    axs[1,2].legend()

    # Noveno gráfico: Espectro vacio
    axs[2,2].plot(df_original_espectro_vacio["E_min"] + 0.5 * (df_original_espectro_vacio["E_max"] - df_original_espectro_vacio["E_min"]), df_original_espectro_vacio["mean"], label="Original")
    axs[2,2].plot(df_sintetico_espectro_vacio["E_min"] + 0.5 * (df_sintetico_espectro_vacio["E_max"] - df_sintetico_espectro_vacio["E_min"]), df_sintetico_espectro_vacio["mean"] * factor_normalizacion, label="Sintético")
    axs[2,2].set_xlabel("Energía [eV]")
    axs[2,2].set_ylabel("Flujo")
    axs[2,2].set_yscale("log")
    axs[2,2].set_xscale("log")  
    axs[2,2].minorticks_on()  # Habilitar marcas menores
    axs[2,2].grid(which="both", linestyle="--", linewidth=0.5)
    axs[2,2].set_title("Espectro vacio")
    axs[2,2].legend()

# Décimo gráfico: Error relativo espectro total
y_data = ((df_original_espectro_total["mean"] - df_sintetico_espectro_total["mean"] * factor_normalizacion) / df_original_espectro_total["mean"] * 100)
axs[0,3].plot(df_original_espectro_total["E_min"] + 0.5 * (df_original_espectro_total["E_max"] - df_original_espectro_total["E_min"]), y_data)
axs[0,3].set_xlabel("Energía [eV]")
axs[0,3].set_ylabel("Error relativo [%]")
axs[0,3].set_xscale("log")
axs[0,3].minorticks_on()  # Habilitar marcas menores
axs[0,3].grid(which="both", linestyle="--", linewidth=0.5)
axs[0,3].set_title("Error relativo espectro total")
axs[0,3].set_ylim(max(y_data.min(), -100), min(y_data.max(), 100))

if geometria[0]:
    # Undécimo gráfico: Error relativo espectro agua
    y_data = ((df_original_espectro_total["mean"] - df_original_espectro_vacio["mean"] - df_sintetico_espectro_total["mean"] * factor_normalizacion + df_sintetico_espectro_vacio["mean"] * factor_normalizacion) / (df_original_espectro_total["mean"] - df_original_espectro_vacio["mean"]) * 100)
    axs[1,3].plot(df_original_espectro_vacio["E_min"] + 0.5 * (df_original_espectro_vacio["E_max"] - df_original_espectro_vacio["E_min"]), y_data)
    axs[1,3].set_xlabel("Energía [eV]")
    axs[1,3].set_ylabel("Error relativo [%]")
    axs[1,3].set_xscale("log")
    axs[1,3].minorticks_on()  # Habilitar marcas menores
    axs[1,3].grid(which="both", linestyle="--", linewidth=0.5)
    axs[1,3].set_title("Error relativo espectro agua")
    axs[1,3].set_ylim(max(y_data.min(), -100), min(y_data.max(), 100))

    # Duodécimo gráfico: Error relativo espectro vacio
    y_data = ((df_original_espectro_vacio["mean"] - df_sintetico_espectro_vacio["mean"] * factor_normalizacion) / df_original_espectro_vacio["mean"] * 100)
    axs[2,3].plot(df_original_espectro_vacio["E_min"] + 0.5 * (df_original_espectro_vacio["E_max"] - df_original_espectro_vacio["E_min"]), y_data)
    axs[2,3].set_xlabel("Energía [eV]")
    axs[2,3].set_ylabel("Error relativo [%]")
    axs[2,3].set_xscale("log")
    axs[2,3].minorticks_on()  # Habilitar marcas menores
    axs[2,3].grid(which="both", linestyle="--", linewidth=0.5)
    axs[2,3].set_title("Error relativo espectro vacio")
    axs[2,3].set_ylim(max(y_data.min(), -100), min(y_data.max(), 100))




# Añadir la información adicional como subtítulo
info_text = f"""
Fuente: {fuente_original}   Geometría: {geometria[1:4]} [cm]    Vacio: {geometria[4:6] if geometria[0] else 'No'}
Posicion superficie: {z0} [cm]  Particulas registradas: {cantidad_registrada:.1e}
N_original: {int(N_original):.1e}   N_sintetico: {int(N_sintetico):.1e}
Columns order: {columns_order}  Binning type: {type}
Micro bins: {micro_bins}    Macro bins: {macro_bins}
"""
fig.suptitle(info_text, fontsize=10)


# Guardar el archivo PNG
plt.savefig("resultados.png")
