import sys

sys.path.append("/home/lucas/Documents/Proyecto_Integrador/PI")
from functions import *

fuente = ["monoenergetica", "colimada"]
geometria = [False, 15, 15, 100, 0.3, 0.3]
z0 = 20
N_particles = 1e6/2

# files_to_remove = ['geometry.xml', 'materials.xml', 'settings.xml', 'tallies.xml', 'original.png','statepoint_original.h5','summary.h5','surface_source.h5','tallies.out','sintetico.png','statepoint_sintetico.h5']

run_simulation(fuente, geometria, z0, int(N_particles))

SurfaceSourceFile = kds.SurfaceSourceFile("surface_source.h5", domain={"w": [0, 2]})
df = SurfaceSourceFile.get_pandas_dataframe()
df = df[["x", "y", "ln(E0/E)", "mu", "phi", "wgt"]]
del SurfaceSourceFile

factor_normalizacion = df["wgt"].sum() / (100 * N_particles)

columns_order = ["ln(E0/E)", "x", "y", "mu", "phi"]
micro_bins = [20000] * len(columns_order)
macro_bins = [15, 10, 8, 6, 5]
N_max = 1e7
type = "equal_area"


bins_for_comparation: int = 100
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
    df, columns_order, micro_bins, macro_bins, type=type
)
del df

df_sampled = sample(cumul, micro, macro, columns_order, int(N_max))

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

fuente = ["sintetico.h5"]

run_simulation(fuente, geometria, z0, int(N_max / 100)) #100 batches


plt.figure()

# Load the statepoint file 1
sp = openmc.StatePoint("statepoint_original.h5")
tally = sp.get_tally(name="flux")
df = tally.get_pandas_dataframe()
df.columns = ["x", "y", "z", "nuclide", "score", "mean", "std.dev."]
plt.plot(df["z"][450:] * 15 / 750, df["mean"][450:], label="Original")

df1 = df

# Load the statepoint file 2
sp = openmc.StatePoint("statepoint_sintetico.h5")
tally = sp.get_tally(name="flux")
df = tally.get_pandas_dataframe()
df.columns = ["x", "y", "z", "nuclide", "score", "mean", "std.dev."]
plt.plot(
    df["z"][450:] * 15 / 750, df["mean"][450:] * factor_normalizacion *3.031/2.278, label="Sintetico"
)
plt.xlabel("z [cm]")
plt.grid()
plt.title("Flux vs. z")
plt.legend()

plt.show()

# error relativo
plt.figure()
plt.plot(
    df1["z"][450:] * 15 / 750,
    (df1["mean"][450:] - df["mean"][450:] * factor_normalizacion) *3.031/2.278/ df1["mean"][450:],
)
plt.xlabel("z [cm]")
plt.grid()
plt.title("Error relativo")
plt.show()
