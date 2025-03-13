import sys

sys.path.append("/home/lucas/Documents/Proyecto_Integrador/PI")
from functions import *

path = "/home/lucas/Documents/Proyecto_Integrador/PI/segundo_semestre/4-3-25/prueba1"
source_path = (
    "/home/lucas/Documents/Proyecto_Integrador/PI/segundo_semestre/sources/source1/"
)
os.chdir(path)

# Inputs:

geometria = [True, 15, 15, 100, 3, 3]
z0 = 30
N_original = int(4.5e8)

fuente_original = ["monoenergetica", "colimada"]
columns_order = ["ln(E0/E)", "x", "y", "mu", "phi"]
micro_bins = [500] * len(columns_order)
# micro_bins = [50000, 40000, 35000,35000,30000]
macro_bins = [10, 8, 8, 7]
N_sintetico = 1e7
N_file = int(
    5e6
)  # toma esta cantidad del archivo fuente, aunque el archivo fuente puede ser mas grande
batches = 30
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

print("flag 1")
df, factor_normalizacion, cantidad_registrada = df_from_source_file(
    source_path + "surface_source.h5", N_original
)

print("flag 2")
cumul, micro, macro = calculate_cumul_micro_macro(
    df,
    columns_order,
    micro_bins,
    macro_bins,
    binning_type=type,
    user_defined_macro_edges=used_defined_edges,
)
del df
print("flag 3")

flujo_total_sintetico = None
flujo_vacio_sintetico = None
espectro_total_sintetico = None
espectro_vacio_sintetico = None


for i in range(batches):
    print(i)
    df_sampled = sample(cumul, micro, macro, columns_order, int(N_sintetico))

    df_formatted = format_df_to_MCPL(df_sampled, z=z0)
    del df_sampled

    kds.create_source_file(df_formatted, "sintetico.h5")
    del df_formatted

    fuente_sintetica = ["sintetico.h5"]

    run_simulation(fuente_sintetica, geometria, z0, int(N_sintetico))

    # Load the statepoint file sintetico por batch
    sp = openmc.StatePoint("statepoint_sintetico.h5")
    tally_total = sp.get_tally(name="flux_total")
    df_sintetico_total = tally_total.get_pandas_dataframe()
    df_sintetico_total.columns = ["x", "y", "z", "nuclide", "score", "mean", "std.dev."]
    aux = df_sintetico_total["mean"].to_numpy()
    if flujo_total_sintetico is None:
        flujo_total_sintetico = np.zeros_like(aux)
    flujo_total_sintetico += aux

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
    aux = df_sintetico_espectro_total["mean"].to_numpy()
    if espectro_total_sintetico is None:
        espectro_total_sintetico = np.zeros_like(aux)
    espectro_total_sintetico += aux

    if geometria[0]:
        tally_vacio = sp.get_tally(name="flux_vacio")
        df_sintetico_vacio = tally_vacio.get_pandas_dataframe()
        df_sintetico_vacio.columns = [
            "x",
            "y",
            "z",
            "nuclide",
            "score",
            "mean",
            "std.dev.",
        ]
        aux = df_sintetico_vacio["mean"].to_numpy()
        if flujo_vacio_sintetico is None:
            flujo_vacio_sintetico = np.zeros_like(aux)
        flujo_vacio_sintetico += aux

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
        aux = df_sintetico_espectro_vacio["mean"].to_numpy()
        if espectro_vacio_sintetico is None:
            espectro_vacio_sintetico = np.zeros_like(aux)
        espectro_vacio_sintetico += aux

    del (
        sp,
        tally_total,
        tally_vacio,
        df_sintetico_total,
        df_sintetico_vacio,
        df_sintetico_espectro_total,
        df_sintetico_espectro_vacio,
    )

flujo_total_sintetico /= batches
flujo_vacio_sintetico /= batches
flujo_agua_sintetico = flujo_total_sintetico - flujo_vacio_sintetico
flujo_total_sintetico /= 37.5
flujo_vacio_sintetico /= 1.5
flujo_agua_sintetico /= 36
espectro_total_sintetico /= batches
espectro_vacio_sintetico /= batches
espectro_agua_sintetico = espectro_total_sintetico - espectro_vacio_sintetico
espectro_total_sintetico /= 225
espectro_vacio_sintetico /= 9
espectro_agua_sintetico /= 216


# Plot the results

# Load the statepoint file 1
sp = openmc.StatePoint(source_path + "statepoint_original.h5")
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
    
    df_original_agua = df_original_total["mean"] - df_original_vacio["mean"]
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
    df_original_espectro_agua = df_original_espectro_total["mean"] - df_original_espectro_vacio["mean"]
    

df_original_total["mean"] /= 37.5
df_original_vacio["mean"] /= 1.5
df_original_agua /= 36
df_original_espectro_total["mean"] /= 225
df_original_espectro_vacio["mean"] /= 9
df_original_espectro_agua /= 216


factor = len(df_original_total) / geometria[3]
z_min = int(z0 * factor)
# z_max = int(
#     max(
#         df_original_total.loc[df_original_total["mean"] != 0, "z"].max(),
#         df_sintetico_total.loc[df_sintetico_total["mean"] != 0, "z"].max(),
#     )
#     * 1.1
# )
z_max = int(geometria[3] * factor)

# Crear los gráficos FLUJO
if geometria[0]:
    fig_flujo = plt.figure(figsize=(16 * 1.25, 9*0.8))
    gs = gridspec.GridSpec(
        2, 3, height_ratios=[2.5, 1]
    )  # La primera fila es el doble de alta que la segunda

    axs_flujo = np.empty((2, 3), dtype=object)  # Para almacenar los ejes como matriz

    for i in range(2):
        for j in range(3):
            axs_flujo[i, j] = fig_flujo.add_subplot(
                gs[i, j]
            )  # Agregar cada subplot según el GridSpec
    # fig_espectro, axs_espectro = plt.subplots(2, 3, figsize=(10, 18))
    plt.subplots_adjust(
        left=0.09, right=0.97, top=0.95, bottom=0.05, wspace=0.4, hspace=0.25
    )
    # plt.tight_layout()


else:
    fig_flujo, axs_flujo = plt.subplots(2, 1, figsize=(10, 6))
    # fig_espectro, axs_espectro = plt.subplots(2, 1, figsize=(10, 6))
    plt.subplots_adjust(top=0.75, wspace=0.4)

axs_flujo = np.atleast_2d(axs_flujo)  # Convierte axs en una matriz 2D siempre
# axs_espectro = np.atleast_2d(axs_espectro)  # Convierte axs en una matriz 2D siempre

# Primer gráfico: Flux vs. z
axs_flujo[0, 0].plot(
    df_original_total["z"][z_min:z_max] / factor,
    df_original_total["mean"][z_min:z_max] ,
    label="Original",
)
axs_flujo[0, 0].plot(
    df_original_total["z"][z_min:z_max] / factor,
    flujo_total_sintetico[z_min:z_max] * factor_normalizacion,
    label="Sintético",
)
axs_flujo[0, 0].set_xlabel("z [cm]")
axs_flujo[0, 0].set_ylabel("$\phi$ [cm$^{-2}$ s$^{-1}$]")
axs_flujo[0, 0].set_yscale("log")
axs_flujo[0, 0].minorticks_on()  # Habilitar marcas menores
axs_flujo[0, 0].grid(which="both", linestyle="--", linewidth=0.5)
axs_flujo[0, 0].set_title("$\phi$ vs. z")
axs_flujo[0, 0].legend()

# Segundo gráfico: Error relativo
y_data = (
    (
        df_original_total["mean"][z_min:]
        - flujo_total_sintetico[z_min:] * factor_normalizacion
    )
    / df_original_total["mean"][z_min:]
    * 100
)
axs_flujo[1, 0].plot(
    df_original_total["z"][z_min:] / factor,
    y_data,
)
# axs_flujo[1, 0].set_xlabel("z [cm]")
axs_flujo[1, 0].set_ylabel("Error relativo [$\%$]")
axs_flujo[1, 0].minorticks_on()  # Habilitar marcas menores
axs_flujo[1, 0].grid(which="both", linestyle="--", linewidth=0.5)
# axs_flujo[1, 0].set_title("Error relativo vs. z")
axs_flujo[1, 0].set_ylim(max(y_data.min(), -100), min(y_data.max(), 100))
axs_flujo[1, 0].axhline(y=0, color="black", linestyle="--", linewidth=1)

if geometria[0]:
    # Tercer gráfico: Flux agua vs. z
    axs_flujo[0, 1].plot(
        df_original_vacio["z"][z_min:z_max] / factor,
        df_original_agua[z_min:z_max],
        label="Original",
    )
    axs_flujo[0, 1].plot(
        df_original_vacio["z"][z_min:z_max] / factor,
        flujo_agua_sintetico[z_min:z_max] * factor_normalizacion,
        label="Sintético",
    )
    axs_flujo[0, 1].set_xlabel("z [cm]")
    axs_flujo[0, 1].set_ylabel("$\phi$ agua [cm$^{-2}$ s$^{-1}$]")
    axs_flujo[0, 1].set_yscale("log")
    axs_flujo[0, 1].minorticks_on()  # Habilitar marcas menores
    axs_flujo[0, 1].grid(which="both", linestyle="--", linewidth=0.5)
    axs_flujo[0, 1].set_title("$\phi$ agua vs. z")
    axs_flujo[0, 1].legend()

    # Cuarto gráfico: Error relativo agua
    y_data = (
        (
            df_original_agua[z_min:]
            - flujo_agua_sintetico[z_min:] * factor_normalizacion
        )
        / (df_original_agua[z_min:])
        * 100
    )
    axs_flujo[1, 1].plot(
        df_original_vacio["z"][z_min:] / factor,
        y_data,
    )
    # axs_flujo[1, 1].set_xlabel("z [cm]")
    axs_flujo[1, 1].set_ylabel("Error relativo [$\%$]")
    axs_flujo[1, 1].minorticks_on()  # Habilitar marcas menores
    axs_flujo[1, 1].grid(which="both", linestyle="--", linewidth=0.5)
    # axs_flujo[1, 1].set_title("Error relativo agua")
    axs_flujo[1, 1].set_ylim(max(y_data.min(), -100), min(y_data.max(), 100))
    axs_flujo[1, 1].axhline(y=0, color="black", linestyle="--", linewidth=1)

    # Quinto gráfico: flux vacio vs. z
    axs_flujo[0, 2].plot(
        df_original_vacio["z"][z_min:z_max] / factor,
        df_original_vacio["mean"][z_min:z_max],
        label="Original",
    )
    axs_flujo[0, 2].plot(
        df_original_vacio["z"][z_min:z_max] / factor,
        flujo_vacio_sintetico[z_min:z_max] * factor_normalizacion,
        label="Sintético",
    )
    axs_flujo[0, 2].set_xlabel("z [cm]")
    axs_flujo[0, 2].set_ylabel("$\phi$ vacio [cm$^{-2}$ s$^{-1}$]")
    axs_flujo[0, 2].set_yscale("log")
    axs_flujo[0, 2].minorticks_on()  # Habilitar marcas menores
    axs_flujo[0, 2].grid(which="both", linestyle="--", linewidth=0.5)
    axs_flujo[0, 2].set_title("$\phi$ vacio vs. z")
    axs_flujo[0, 2].legend()

    # Sexto gráfico: Error relativo vacio
    y_data = (
        (
            df_original_vacio["mean"][z_min:]
            - flujo_vacio_sintetico[z_min:] * factor_normalizacion
        )
        / df_original_vacio["mean"][z_min:]
        * 100
    )
    axs_flujo[1, 2].plot(
        df_original_vacio["z"][z_min:] / factor,
        y_data,
    )
    # axs_flujo[1, 2].set_xlabel("z [cm]")
    axs_flujo[1, 2].set_ylabel("Error relativo [$\%$]")
    axs_flujo[1, 2].minorticks_on()  # Habilitar marcas menores
    axs_flujo[1, 2].grid(which="both", linestyle="--", linewidth=0.5)
    # axs_flujo[1, 2].set_title("Error relativo vacio")
    axs_flujo[1, 2].set_ylim(max(y_data.min(), -100), min(y_data.max(), 100))
    axs_flujo[1, 2].axhline(y=0, color="black", linestyle="--", linewidth=1)

# Guardar el archivo PNG
plt.savefig("resultados_flujo.png")

# Crear los gráficos ESPECTRO
if geometria[0]:
    fig_espectro = plt.figure(figsize=(16*1.25, 9*0.8))
    gs = gridspec.GridSpec(
        2, 3, height_ratios=[2.5, 1]
    )  # La primera fila es el doble de alta que la segunda

    axs_espectro = np.empty((2, 3), dtype=object)  # Para almacenar los ejes como matriz

    for i in range(2):
        for j in range(3):
            axs_espectro[i, j] = fig_espectro.add_subplot(
                gs[i, j]
            )  # Agregar cada subplot según el GridSpec
    plt.subplots_adjust(
        left=0.09, right=0.97, top=0.95, bottom=0.05, wspace=0.4, hspace=0.25
    )
    # plt.tight_layout()


else:
    fig_espectro, axs_espectro = plt.subplots(2, 1, figsize=(10, 6))
    # fig_espectro, axs_espectro = plt.subplots(2, 1, figsize=(10, 6))
    plt.subplots_adjust(top=0.75, wspace=0.4)

axs_espectro = np.atleast_2d(axs_espectro)  # Convierte axs en una matriz 2D siempre

# Séptimo gráfico: Espectro total
axs_espectro[0, 0].plot(
    df_original_espectro_total["E_min"]
    + 0.5 * (df_original_espectro_total["E_max"] - df_original_espectro_total["E_min"]),
    df_original_espectro_total["mean"],
    label="Original",
)
axs_espectro[0, 0].plot(
    df_original_espectro_total["E_min"]
    + 0.5 * (df_original_espectro_total["E_max"] - df_original_espectro_total["E_min"]),
    espectro_total_sintetico * factor_normalizacion,
    label="Sintético",
)
axs_espectro[0, 0].set_xlabel("Energía [eV]")
axs_espectro[0, 0].set_ylabel("$\phi$ [cm$^{-2}$ s$^{-1}$ eV$^{-1}$]")
axs_espectro[0, 0].set_yscale("log")
axs_espectro[0, 0].set_xscale("log")
axs_espectro[0, 0].minorticks_on()  # Habilitar marcas menores
axs_espectro[0, 0].grid(which="both", linestyle="--", linewidth=0.5)
axs_espectro[0, 0].set_title("Espectro total")
axs_espectro[0, 0].legend()

if geometria[0]:
    # Octavo gráfico: Espectro agua
    axs_espectro[0, 1].plot(
        df_original_espectro_vacio["E_min"]
        + 0.5
        * (df_original_espectro_vacio["E_max"] - df_original_espectro_vacio["E_min"]),
        df_original_espectro_agua,
        label="Original",
    )
    axs_espectro[0, 1].plot(
        df_original_espectro_vacio["E_min"]
        + 0.5
        * (df_original_espectro_vacio["E_max"] - df_original_espectro_vacio["E_min"]),
        espectro_agua_sintetico * factor_normalizacion,
        label="Sintético",
    )
    axs_espectro[0, 1].set_xlabel("Energía [eV]")
    axs_espectro[0, 1].set_ylabel("$\phi$ agua [cm$^{-2}$ s$^{-1}$ eV$^{-1}$]")
    axs_espectro[0, 1].set_yscale("log")
    axs_espectro[0, 1].set_xscale("log")
    axs_espectro[0, 1].minorticks_on()  # Habilitar marcas menores
    axs_espectro[0, 1].grid(which="both", linestyle="--", linewidth=0.5)
    axs_espectro[0, 1].set_title("Espectro agua")
    axs_espectro[0, 1].legend()

    # Noveno gráfico: Espectro vacio
    axs_espectro[0, 2].plot(
        df_original_espectro_vacio["E_min"]
        + 0.5
        * (df_original_espectro_vacio["E_max"] - df_original_espectro_vacio["E_min"]),
        df_original_espectro_vacio["mean"],
        label="Original",
    )
    axs_espectro[0, 2].plot(
        df_original_espectro_vacio["E_min"]
        + 0.5
        * (df_original_espectro_vacio["E_max"] - df_original_espectro_vacio["E_min"]),
        espectro_vacio_sintetico * factor_normalizacion,
        label="Sintético",
    )
    axs_espectro[0, 2].set_xlabel("Energía [eV]")
    axs_espectro[0, 2].set_ylabel("$\phi$ vacio [cm$^{-2}$ s$^{-1}$ eV$^{-1}$]")
    axs_espectro[0, 2].set_yscale("log")
    axs_espectro[0, 2].set_xscale("log")
    axs_espectro[0, 2].minorticks_on()  # Habilitar marcas menores
    axs_espectro[0, 2].grid(which="both", linestyle="--", linewidth=0.5)
    axs_espectro[0, 2].set_title("Espectro vacio")
    axs_espectro[0, 2].legend()

# Décimo gráfico: Error relativo espectro total
y_data = (
    (
        df_original_espectro_total["mean"]
        - espectro_total_sintetico * factor_normalizacion
    )
    / df_original_espectro_total["mean"]
    * 100
)
axs_espectro[1, 0].plot(
    df_original_espectro_total["E_min"]
    + 0.5 * (df_original_espectro_total["E_max"] - df_original_espectro_total["E_min"]),
    y_data,
)
# axs_espectro[1, 0].set_xlabel("Energía [eV]")
axs_espectro[1, 0].set_ylabel("Error relativo [$\%$]")
axs_espectro[1, 0].set_xscale("log")
axs_espectro[1, 0].minorticks_on()  # Habilitar marcas menores
axs_espectro[1, 0].grid(which="both", linestyle="--", linewidth=0.5)
# axs_espectro[1, 0].set_title("Error relativo espectro total")
axs_espectro[1, 0].set_ylim(max(y_data.min(), -100), min(y_data.max(), 100))
axs_espectro[1, 0].axhline(y=0, color="black", linestyle="--", linewidth=1)

if geometria[0]:
    # Undécimo gráfico: Error relativo espectro agua
    y_data = (
        (
            df_original_espectro_agua
            - espectro_agua_sintetico * factor_normalizacion
        )
        / (df_original_espectro_total["mean"] - df_original_espectro_vacio["mean"])
        * 100
    )
    axs_espectro[1, 1].plot(
        df_original_espectro_vacio["E_min"]
        + 0.5
        * (df_original_espectro_vacio["E_max"] - df_original_espectro_vacio["E_min"]),
        y_data,
    )
    # axs_espectro[1, 1].set_xlabel("Energía [eV]")
    axs_espectro[1, 1].set_ylabel("Error relativo [$\%$]")
    axs_espectro[1, 1].set_xscale("log")
    axs_espectro[1, 1].minorticks_on()  # Habilitar marcas menores
    axs_espectro[1, 1].grid(which="both", linestyle="--", linewidth=0.5)
    # axs_espectro[1, 1].set_title("Error relativo espectro agua")
    axs_espectro[1, 1].set_ylim(max(y_data.min(), -100), min(y_data.max(), 100))
    axs_espectro[1, 1].axhline(y=0, color="black", linestyle="--", linewidth=1)

    # Duodécimo gráfico: Error relativo espectro vacio
    y_data = (
        (
            df_original_espectro_vacio["mean"]
            - espectro_vacio_sintetico * factor_normalizacion
        )
        / df_original_espectro_vacio["mean"]
        * 100
    )
    axs_espectro[1, 2].plot(
        df_original_espectro_vacio["E_min"]
        + 0.5
        * (df_original_espectro_vacio["E_max"] - df_original_espectro_vacio["E_min"]),
        y_data,
    )
    # axs_espectro[1, 2].set_xlabel("Energía [eV]")
    axs_espectro[1, 2].set_ylabel("Error relativo [$\%$]")
    axs_espectro[1, 2].set_xscale("log")
    axs_espectro[1, 2].minorticks_on()  # Habilitar marcas menores
    axs_espectro[1, 2].grid(which="both", linestyle="--", linewidth=0.5)
    # axs_espectro[1, 2].set_title("Error relativo espectro vacio")
    axs_espectro[1, 2].set_ylim(max(y_data.min(), -100), min(y_data.max(), 100))
    axs_espectro[1, 2].axhline(y=0, color="black", linestyle="--", linewidth=1)

# Guardar el archivo PNG
plt.savefig("resultados_espectro.png")


# # Añadir la información adicional como subtítulo
# info_text = f"""
# Fuente: {fuente_original}   Geometría: {geometria[1:4]} [cm]    Vacio: {geometria[4:6] if geometria[0] else 'No'}
# Posicion superficie: {z0} [cm]  Particulas registradas: {cantidad_registrada:.1e}
# N_original: {int(N_original):.1e}   N_sintetico: {int(N_sintetico):.1e}
# Columns order: {columns_order}  Binning type: {type}
# Micro bins: {micro_bins}    Macro bins: {macro_bins}
# """
# fig.suptitle(info_text, fontsize=10)
