import sys

sys.path.append("/home/lucas/Documents/Proyecto_Integrador/PI")
from functions import *

path = "/home/lucas/Documents/Proyecto_Integrador/PI/segundo_semestre/5-2-25"
os.chdir(path)

fuente_original = ["monoenergetica", "colimada"]
geometria = [True, 15, 15, 100, 3, 3]
z0 = 5
N_original = 1e8 / 2 / 5/2

# files_to_remove = ['geometry.xml', 'materials.xml', 'settings.xml', 'tallies.xml', 'original.png','statepoint_original.h5','summary.h5','surface_source.h5','tallies.out','sintetico.png','statepoint_sintetico.h5']

run_simulation(fuente_original, geometria, z0, int(N_original))

SurfaceSourceFile = kds.SurfaceSourceFile("surface_source.h5", domain={"w": [0, 2]})
df = SurfaceSourceFile.get_pandas_dataframe()
df = df[["x", "y", "ln(E0/E)", "mu", "phi", "wgt"]]
del SurfaceSourceFile

factor_normalizacion = df["wgt"].sum() / N_original
cantidad_registrada = len(df)

columns_order = ["ln(E0/E)", "x", "y", "mu", "phi"]
micro_bins = [60000] * len(columns_order)
macro_bins = [20, 10, 8, 6]
N_sintetico = 1e7 / 2/2/2
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

cumul, micro, macro = calculate_cumul_micro_macro(
    df, columns_order, micro_bins, macro_bins, binning_type=type
)
del df

df_sampled = sample(cumul, micro, macro, columns_order, int(N_sintetico))

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


df_formatted = format_df_to_MCPL(df_sampled, z=z0)
del df_sampled


kds.create_source_file(df_formatted, "sintetico.h5")

fuente_sintetica = ["sintetico.h5"]

run_simulation(fuente_sintetica, geometria, z0, int(N_sintetico))

# Plot the results

# Load the statepoint file 1
sp = openmc.StatePoint("statepoint_original.h5")
tally_total = sp.get_tally(name="flux_total")
df_original_total = tally_total.get_pandas_dataframe()
df_original_total.columns = ["x", "y", "z", "nuclide", "score", "mean", "std.dev."]
if geometria[0]:
    tally_vacio = sp.get_tally(name="flux_vacio")
    df_original_vacio = tally_vacio.get_pandas_dataframe()
    df_original_vacio.columns = ["x", "y", "z", "nuclide", "score", "mean", "std.dev."]

# Load the statepoint file 2
sp = openmc.StatePoint("statepoint_sintetico.h5")
tally_total = sp.get_tally(name="flux_total")
df_sintetico_total = tally_total.get_pandas_dataframe()
df_sintetico_total.columns = ["x", "y", "z", "nuclide", "score", "mean", "std.dev."]
if geometria[0]:
    tally_vacio = sp.get_tally(name="flux_vacio")
    df_sintetico_vacio = tally_vacio.get_pandas_dataframe()
    df_sintetico_vacio.columns = ["x", "y", "z", "nuclide", "score", "mean", "std.dev."]

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
    fig, axs = plt.subplots(2, 2, figsize=(12, 6))
    plt.subplots_adjust(
        left=0.1, right=0.9, top=0.75, bottom=0.1, wspace=0.4, hspace=0.4
    )

else:
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
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
axs[0, 1].plot(
    df_original_total["z"][z_min:] / factor,
    (
        df_original_total["mean"][z_min:]
        - df_sintetico_total["mean"][z_min:] * factor_normalizacion
    )
    / df_original_total["mean"][z_min:]
    * 100,
)
axs[0, 1].set_xlabel("z [cm]")
axs[0, 1].set_ylabel("Error relativo [%]")
axs[0, 1].minorticks_on()  # Habilitar marcas menores
axs[0, 1].grid(which="both", linestyle="--", linewidth=0.5)
axs[0, 1].set_title("Error relativo vs. z")

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

    # Cuarto gráfico: flux vacio vs. z
    axs[1, 1].plot(
        df_original_vacio["z"][z_min:z_max] / factor,
        df_original_vacio["mean"][z_min:z_max],
        label="Original",
    )
    axs[1, 1].plot(
        df_sintetico_vacio["z"][z_min:z_max] / factor,
        df_sintetico_vacio["mean"][z_min:z_max] * factor_normalizacion,
        label="Sintético",
    )
    axs[1, 1].set_xlabel("z [cm]")
    axs[1, 1].set_ylabel("Flux vacio")
    axs[1, 1].set_yscale("log")
    axs[1, 1].minorticks_on()  # Habilitar marcas menores
    axs[1, 1].grid(which="both", linestyle="--", linewidth=0.5)
    axs[1, 1].set_title("Flux vacio vs. z")
    axs[1, 1].legend()


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
