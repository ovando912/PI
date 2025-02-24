import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import kdsource as kds
import time
import os
import math
from PIL import Image, ImageDraw, ImageFont
from scipy.stats import entropy
from scipy.stats import wasserstein_distance
from scipy.special import rel_entr

import glob
import shutil
import openmc

plt.rcParams.update({
    "text.usetex": True,  # Usa LaTeX para el texto
    "font.family": "serif",  # Fuente tipo serif (similar a LaTeX)
    "font.serif": ["Computer Modern Roman"],  # Usa Computer Modern
    "axes.labelsize": 20,  # Tamaño de etiquetas de ejes
    "font.size": 20,  # Tamaño general de fuente
    "legend.fontsize": 18,  # Tamaño de fuente en la leyenda
    "xtick.labelsize": 18,  # Tamaño de fuente en los ticks de x
    "ytick.labelsize": 18,  # Tamaño de fuente en los ticks de y
})

# %matplotlib widget


def function(df1: pd.DataFrame, df2: pd.DataFrame):
    # Parametros
    columns_order = ["ln(E0/E)", "x", "y", "mu", "phi"]
    micro_bins = [1000] * len(columns_order)
    macro_bins = [6, 3, 3, 3]
    type = "equal_bins"

    cumul, micro, macro = calculate_cumul_micro_macro(df1, columns_order, micro_bins, macro_bins, type)

    extremos_variables = []
    for column in columns_order:
        extremos_variables.append([df1[column].min(), df1[column].max()])
    
    bins = 100
    resultados = []
    for i in range(len(columns_order)):
        resultados.append([])
        for _ in columns_order:
            resultados[i].append([])

    
    for i in range(len(columns_order)):
        for j in range(i+1):
            if i == j:
                resultados[i][j] = np.zeros(bins)
            else:
                resultados[i][j] = np.zeros((bins, bins))

    




def df_from_source_file(source_file: str, N_original: int, df_size: int = None) -> tuple[pd.DataFrame, float, int]:
    SurfaceSourceFile = kds.SurfaceSourceFile(
        source_file, domain={"w": [0, 2]}
    )
    df = SurfaceSourceFile.get_pandas_dataframe()
    df = df[["x", "y", "ln(E0/E)", "mu", "phi", "wgt"]]
    del SurfaceSourceFile

    if df_size is not None:
        if df_size < len(df):
            N_original = N_original * df_size / len(df)
            base = np.random.randint(0, len(df) - df_size)
            df = df.iloc[base : base + df_size]

    factor_normalizacion = df["wgt"].sum() / N_original
    cantidad_registrada = len(df)
    return df, factor_normalizacion, int(cantidad_registrada)


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
    df_result = pd.DataFrame(
        {
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
        }
    )

    # Ensure integer columns are correctly typed
    df_result["id"] = df_result["id"].astype(int)
    df_result["type"] = df_result["type"].astype(int)
    df_result["userflags"] = df_result["userflags"].astype(int)

    return df_result


def run_simulation(fuente: list, geometria: list, z0: float, N_particles: int) -> None:
    """
    Ejecuta una simulación en OpenMC con la configuración especificada.

    Parámetros:
      fuente (list):
          - Si tiene 1 elemento: se asume que es la ruta a un archivo de fuente.
          - Si tiene 2 elementos: se asume que es [fuente_energia, fuente_direccion] para fuente independiente.
              - fuente_energia puede ser:
                  - "monoenergetica": Fuente con energía fija.
                  - "espectro_fision": Fuente con espectro de fisión.
                  - "espectro_termico": Fuente con espectro térmico.
              - fuente_direccion puede ser:
                  - "colimada": Fuente con dirección fija.
                  - "isotropica": Fuente con distribución isotrópica.

      geometria (list): Parámetros geométricos:
          - geometria[0] (bool): Flag para indicar si existe región de vacío.
          - geometria[1] (float): L_x, ancho del paralelepípedo.
          - geometria[2] (float): L_y, ancho del paralelepípedo.
          - geometria[3] (float): L_z, altura del paralelepípedo.
          - Si geometria[0] es True, se esperan además:
                - geometria[4] (float): L_x_vacio.
                - geometria[5] (float): L_y_vacio.

      z0 (float): Posición en z para la superficie de track.
      N_particles (int): Número de partículas a simular en total.
    """

    # Configuración de secciones de reacción
    openmc.config["cross_sections"] = (
        "/home/lucas/Documents/Proyecto_Integrador/endfb-viii.0-hdf5/cross_sections.xml"
    )

    # --------------------------------------------------------------------------
    # Procesamiento de la fuente
    # --------------------------------------------------------------------------
    if len(fuente) == 1:
        source_file = fuente[0]
        source = openmc.FileSource(source_file)
    elif len(fuente) == 2:
        fuente_energia, fuente_direccion = fuente
        source = openmc.IndependentSource()
        source.particle = "neutron"

        # Distribución espacial: se coloca en la región central de la geometría
        L_x, L_y = geometria[1], geometria[2]
        x_dist = openmc.stats.Uniform(-L_x / 2, L_x / 2)
        y_dist = openmc.stats.Uniform(-L_y / 2, L_y / 2)
        z_dist = openmc.stats.Discrete(1e-6, 1)  # Se fija z muy cerca de 0
        source.space = openmc.stats.CartesianIndependent(x_dist, y_dist, z_dist)

        # Distribución de energía
        if fuente_energia == "monoenergetica":
            source.energy = openmc.stats.Discrete([1e6], [1])

        # Distribución angular
        if fuente_direccion == "colimada":
            mu = openmc.stats.Discrete([1], [1])
            phi = openmc.stats.Uniform(0.0, 2 * np.pi)
            source.angle = openmc.stats.PolarAzimuthal(mu, phi)
    else:
        raise ValueError("El parámetro 'fuente' debe contener 1 o 2 elementos.")

    # --------------------------------------------------------------------------
    # Procesamiento de la geometría
    # --------------------------------------------------------------------------
    # Extraer parámetros geométricos
    vacio = geometria[0]
    L_x, L_y, L_z = geometria[1:4]
    if vacio:
        L_x_vacio, L_y_vacio = geometria[4:6]

    # Definir material: agua
    mat_agua = openmc.Material(name="agua")
    mat_agua.add_nuclide("H1", 2.0, "ao")
    mat_agua.add_nuclide("O16", 1.0, "ao")
    mat_agua.add_s_alpha_beta("c_H_in_H2O")
    mat_agua.set_density("g/cm3", 1.0)
    mats = openmc.Materials([mat_agua])
    mats.export_to_xml()

    # Definir superficies externas
    surfaces = {
        "x_min": openmc.XPlane(x0=-L_x / 2, boundary_type="vacuum"),
        "x_max": openmc.XPlane(x0=L_x / 2, boundary_type="vacuum"),
        "y_min": openmc.YPlane(y0=-L_y / 2, boundary_type="vacuum"),
        "y_max": openmc.YPlane(y0=L_y / 2, boundary_type="vacuum"),
        "z_min": openmc.ZPlane(z0=0, boundary_type="vacuum"),
        "z_max": openmc.ZPlane(z0=L_z, boundary_type="vacuum"),
        "z_track": openmc.ZPlane(z0=z0, boundary_type="transmission", surface_id=70),
    }

    # Si hay vacío, definir superficies internas
    if vacio:
        surfaces.update(
            {
                "x_min_vacio": openmc.XPlane(
                    x0=-L_x_vacio / 2, boundary_type="transmission"
                ),
                "x_max_vacio": openmc.XPlane(
                    x0=L_x_vacio / 2, boundary_type="transmission"
                ),
                "y_min_vacio": openmc.YPlane(
                    y0=-L_y_vacio / 2, boundary_type="transmission"
                ),
                "y_max_vacio": openmc.YPlane(
                    y0=L_y_vacio / 2, boundary_type="transmission"
                ),
            }
        )

    # Para fuente tipo FileSource se traduce la superficie inferior para posicionar z0
    if len(fuente) == 1:
        surfaces["z_min"].translate(vector=(0, 0, z0 - 1e-6), inplace=True)

    # Definir regiones
    region_externa = (
        +surfaces["x_min"]
        & -surfaces["x_max"]
        & +surfaces["y_min"]
        & -surfaces["y_max"]
        & +surfaces["z_min"]
        & -surfaces["z_max"]
    )

    if vacio:
        region_vacio = (
            +surfaces["x_min_vacio"]
            & -surfaces["x_max_vacio"]
            & +surfaces["y_min_vacio"]
            & -surfaces["y_max_vacio"]
            & +surfaces["z_min"]
            & -surfaces["z_max"]
        )

    # Crear universo y definir celdas según configuración de fuente y vacío
    universe = openmc.Universe()

    if vacio:
        if len(fuente) == 2:
            universe.add_cell(
                openmc.Cell(
                    region=region_externa & ~region_vacio & -surfaces["z_track"],
                    fill=mat_agua,
                    name="agua1",
                )
            )
            universe.add_cell(
                openmc.Cell(
                    region=region_externa & ~region_vacio & +surfaces["z_track"],
                    fill=mat_agua,
                    name="agua2",
                )
            )
            universe.add_cell(
                openmc.Cell(
                    region=region_vacio & -surfaces["z_track"], fill=None, name="vacio1"
                )
            )
            universe.add_cell(
                openmc.Cell(
                    region=region_vacio & +surfaces["z_track"], fill=None, name="vacio2"
                )
            )
        else:
            universe.add_cell(
                openmc.Cell(
                    region=region_externa & ~region_vacio, fill=mat_agua, name="agua"
                )
            )
            universe.add_cell(openmc.Cell(region=region_vacio, fill=None, name="vacio"))
    else:
        if len(fuente) == 2:
            universe.add_cell(
                openmc.Cell(
                    region=region_externa & -surfaces["z_track"],
                    fill=mat_agua,
                    name="agua1",
                )
            )
            universe.add_cell(
                openmc.Cell(
                    region=region_externa & +surfaces["z_track"],
                    fill=mat_agua,
                    name="agua2",
                )
            )
        else:
            universe.add_cell(
                openmc.Cell(region=region_externa, fill=mat_agua, name="agua")
            )

    geom = openmc.Geometry(universe)
    geom.export_to_xml()

    # --------------------------------------------------------------------------
    # Configuración de settings y tallies
    # --------------------------------------------------------------------------
    settings = openmc.Settings()
    if len(fuente) == 2:
        settings.surf_source_write = {"surface_ids": [70], "max_particles": 20000000}
    settings.run_mode = "fixed source"
    settings.batches = 100
    settings.particles = int(N_particles / 100)
    settings.source = source
    settings.export_to_xml()

    # Tally: malla para flujo total
    mesh = openmc.RectilinearMesh()
    mesh.x_grid = np.linspace(-L_x / 2, L_x / 2, 2)
    mesh.y_grid = np.linspace(-L_y / 2, L_y / 2, 2)
    mesh.z_grid = np.linspace(0, L_z, 601)
    mesh_filter = openmc.MeshFilter(mesh)
    tally_flux_total = openmc.Tally(name="flux_total")
    tally_flux_total.filters = [mesh_filter]
    tally_flux_total.scores = ["flux"]

    # Tally: malla para flujo en vacio
    if vacio:
        mesh_vacio = openmc.RectilinearMesh()
        mesh_vacio.x_grid = np.linspace(-L_x_vacio / 2, L_x_vacio / 2, 2)
        mesh_vacio.y_grid = np.linspace(-L_y_vacio / 2, L_y_vacio / 2, 2)
        mesh_vacio.z_grid = np.linspace(0, L_z, 601)
        mesh_filter_vacio = openmc.MeshFilter(mesh_vacio)
        tally_flux_vacio = openmc.Tally(name="flux_vacio")
        tally_flux_vacio.filters = [mesh_filter_vacio]
        tally_flux_vacio.scores = ["flux"]

    tallies = openmc.Tallies(
        [tally_flux_total, tally_flux_vacio] if vacio else [tally_flux_total]
    )

    # Tally: superficie para espectro en vacio

    mesh_total = openmc.RectilinearMesh()
    mesh_total.x_grid = np.linspace(-L_x / 2, L_x / 2, 2)
    mesh_total.y_grid = np.linspace(-L_y / 2, L_y / 2, 2)
    mesh_total.z_grid = np.linspace(L_z * 0.99, L_z, 2)
    if vacio:
        mesh_vacio = openmc.RectilinearMesh()
        mesh_vacio.x_grid = np.linspace(-L_x_vacio / 2, L_x_vacio / 2, 2)
        mesh_vacio.y_grid = np.linspace(-L_y_vacio / 2, L_y_vacio / 2, 2)
        mesh_vacio.z_grid = np.linspace(L_z * 0.99, L_z, 2)

    tally_surface = openmc.Tally(name="espectro_total")
    tally_surface.filters = [openmc.MeshFilter(mesh_total),openmc.EnergyFilter(np.logspace(-3, 7, 75))]
    tally_surface.scores = ["flux"]
    tallies.append(tally_surface)

    if vacio:
        tally_surface = openmc.Tally(name="espectro_vacio")
        tally_surface.filters = [openmc.MeshFilter(mesh_vacio),openmc.EnergyFilter(np.logspace(-3, 7, 75))]
        tally_surface.scores = ["flux"]
        tallies.append(tally_surface)

    tallies.export_to_xml()

    # --------------------------------------------------------------------------
    # Limpieza de archivos previos y ejecución de la simulación
    # --------------------------------------------------------------------------
    for file in glob.glob("statepoint.*.h5"):
        os.remove(file)
    if os.path.exists("summary.h5"):
        os.remove("summary.h5")

    openmc.run()

    # Mover archivos de salida según tipo de fuente
    statepoint_files = glob.glob("statepoint.*.h5")
    nuevo_nombre = (
        "statepoint_original.h5" if len(fuente) == 2 else "statepoint_sintetico.h5"
    )
    for file in statepoint_files:
        shutil.move(file, nuevo_nombre)


def calculate_cumul_micro_macro(
    df: pd.DataFrame,
    columns: list,
    micro_bins: list,
    macro_bins: list,
    binning_type: str = "equal_bins",
    user_defined_macro_edges: list = None,
) -> tuple[list, list, list]:
    """
    Calcula recursivamente los histogramas acumulados (cumulative) para cada dimensión.

    Parámetros:
      df (pd.DataFrame): DataFrame con los datos.
      columns (list): Lista con los nombres de las columnas a procesar.
      micro_bins (list): Lista con el número de bins para el histograma acumulado de la columna micro.
      macro_bins (list): Lista con el número de bins para el histograma acumulado de la columna macro.
      binning_type (str): Tipo de binning a utilizar ('equal_bins' o 'equal_area').

    Retorna:
      Tuple de tres listas:
        - cumul_list: Histogramas acumulados para cada dimensión.
        - micro_list: Límites de los bins micro para cada dimensión.
        - macro_list: Límites de los bins macro para cada dimensión (None para la última dimensión). #TODO sacar None
    """
    # -----------------------------------------------------------------------------
    # 1. Determinación de casos triviales
    # -----------------------------------------------------------------------------
    if len(df) == 0:
        auxiliar = []
        for i in range(len(columns)):
            elemento = None
            for _ in range(i):
                elemento = [elemento]
            auxiliar.append(elemento)
        return auxiliar, auxiliar, auxiliar

    min_df, max_df = min(df[columns[0]]), max(df[columns[0]])

    if min_df == max_df:
        # Caso base: si solo queda una columna, no se define macro (se termina la recursión)
        if len(columns) == 1:
            macro_edges = None
        else:
            macro_edges = np.array([min_df - 1, min_df + 1])
        micro_edges = np.array([min_df])
        cumul = np.array([1])

    # -----------------------------------------------------------------------------
    # 2. A partir de aca el df tiene por lo menos 2 valores distintos
    # -----------------------------------------------------------------------------
    else:
        # Caso base: si solo queda una columna, no se define macro (se termina la recursión)
        if len(columns) == 1:
            macro_edges = None

    # -----------------------------------------------------------------------------
    # 3. Determinación de los bins macro para la columna actual según el tipo de binning
    # -----------------------------------------------------------------------------
        elif binning_type == "equal_bins":
            if user_defined_macro_edges[0] is not None:
                macro_edges_aux = (
                    [min_df]
                    + [x for x in user_defined_macro_edges[0] if min_df < x < max_df]
                    + [max_df]
                )
            else:
                macro_edges_aux = [min_df] + [max_df]
            macro_edges_width = np.diff(macro_edges_aux)
            macro_edges_width = (
                macro_edges_width / macro_edges_width.sum() * macro_bins[0]
            )
            macro_edges_width = np.array(
                [math.ceil(x - 0.1) for x in macro_edges_width]
            )

            macro_edges = []
            for i in range(len(macro_edges_aux) - 1):
                start = macro_edges_aux[i]
                end = macro_edges_aux[i + 1]
                # Genera los puntos del segmento.
                seg = np.linspace(start, end, macro_edges_width[i] + 2)
                # Si no es el primer segmento, se omite el primer valor para evitar duplicados.
                if i > 0:
                    seg = seg[1:]
                macro_edges.append(seg)

            # Concatena todos los segmentos en un solo array
            macro_edges = np.concatenate(macro_edges)

            # Deleteo variables inecesarias
            del macro_edges_aux, macro_edges_width, start, end, seg, min_df, max_df

        elif binning_type == "equal_area":
            # Usando pd.qcut para obtener cortes que dividan en áreas iguales
            _, macro_edges = pd.qcut(
                df[columns[0]],
                q=macro_bins[0],
                labels=False,
                retbins=True,
                duplicates="drop",
            )

        else:
            raise ValueError(
                "El parámetro 'binning_type' debe ser 'equal_bins' o 'equal_area'."
            )

    # -----------------------------------------------------------------------------
    # 4. Procesamiento micro
    # -----------------------------------------------------------------------------
        micro_edges = np.linspace(
            min(df[columns[0]]), max(df[columns[0]]), micro_bins[0] + 1
        )
        if len(columns) > 1:
            micro_edges = np.array(sorted(set(np.concatenate((micro_edges, macro_edges[1: -1])))))
        # Calcula el histograma ponderado (usando la columna "wgt") para la columna actual
        counts, _ = np.histogram(df[columns[0]], bins=micro_edges, weights=df["wgt"])

        # Calcula el histograma acumulado normalizado
        cumul = np.insert(np.cumsum(counts) / counts.sum(), 0, 0)

        # Encontrar dónde cambian los valores
        diffs = np.diff(cumul)
        change_indices = np.where(diffs != 0)[0]  # Índices donde cambia

        # Agregar primeros y últimos índices de cada grupo
        first_indices = np.insert(change_indices + 1, 0, 0)  # Primeros de cada grupo
        last_indices = np.append(change_indices, len(cumul) - 1)  # Últimos de cada grupo

        # Unir y ordenar los índices
        selected_indices = np.unique(np.concatenate((first_indices, last_indices)))

        # Filtrar los arrays
        micro_edges = micro_edges[selected_indices]
        cumul = cumul[selected_indices]

    # Inicializa las listas para guardar resultados
    cumul_list = [cumul]
    micro_list = [micro_edges]
    macro_list = [macro_edges]

    if len(columns) == 1:
        return cumul_list, micro_list, macro_list

    # -----------------------------------------------------------------------------
    # 3. División del DataFrame según los bins macro para aplicar recursividad
    # -----------------------------------------------------------------------------
    # Asigna cada fila del DataFrame al bin correspondiente de acuerdo a macro_edges
    bin_indices = np.digitize(df[columns[0]], bins=macro_edges) - 1

    # Prepara las listas para almacenar resultados recursivos en cada subgrupo (cada bin macro)
    # Se reserva un espacio en cada lista para cada dimensión adicional
    for _ in range(len(columns) - 1):
        cumul_list.append([])
        micro_list.append([])
        macro_list.append([])

    # Procesa recursivamente cada subgrupo definido por los bins macro
    for bin_idx in range(len(macro_edges) - 1):
        # Filtra las filas que caen en el bin actual
        if bin_idx == len(macro_edges) - 2:
            # Última iteración: incluir bin_idx y bin_idx+1
            df_filtered = df[(bin_indices == bin_idx) | (bin_indices == bin_idx + 1)]
        else:
            df_filtered = df[bin_indices == bin_idx]

        # Llama recursivamente para las columnas restantes
        cumul_aux, micro_aux, macro_aux = calculate_cumul_micro_macro(
            df_filtered,
            columns[1:],
            micro_bins[1:],
            macro_bins[1:],
            binning_type,
            user_defined_macro_edges[1:],
        )

        # Guarda los resultados recursivos en las listas correspondientes
        for j in range(len(cumul_aux)):
            cumul_list[j + 1].append(cumul_aux[j])
            micro_list[j + 1].append(micro_aux[j])
            macro_list[j + 1].append(macro_aux[j])

    return cumul_list, micro_list, macro_list


def index_management(data: np.ndarray, indices: list) -> np.ndarray:
    """
    Get the value of a multi-dimensional array using a list of indices. Used in the sample function.

    Parameter:
    - data: Multi-dimensional array.
    - indices: List of indices.

    Return:
    - The value of the array at the specified indices.
    """
    value = data
    for i in indices:
        value = value[i]
    return value


def sample(
    cumul: list, micro: list, macro: list, columns: list, N: int
) -> pd.DataFrame:
    """
    Sample N values from a N-dimensional distribution.

    Parameter:
    - cumul: List with the cumulative histograms for each dimension.
    - micro: List with the micro histograms for each dimension.
    - macro: List with the macro histograms for each dimension.
    - columns: List with the names of the columns.
    - N: Number of samples to generate.

    Return:
    - A DataFrame with the sampled values.
    """
    sampled_values = []

    if len(columns) == 6:
        for _ in range(N):
            # First dimension
            sampled_0 = np.interp(np.random.rand(), cumul[0], micro[0])
            index_0 = np.searchsorted(macro[0], sampled_0) - 1

            # Second dimension
            sampled_1 = np.interp(
                np.random.rand(), cumul[1][index_0], micro[1][index_0]
            )
            index_1 = np.searchsorted(macro[1][index_0], sampled_1) - 1

            # Third dimension
            sampled_2 = np.interp(
                np.random.rand(), cumul[2][index_0][index_1], micro[2][index_0][index_1]
            )
            index_2 = np.searchsorted(macro[2][index_0][index_1], sampled_2) - 1

            # Fourth dimension
            sampled_3 = np.interp(
                np.random.rand(),
                cumul[3][index_0][index_1][index_2],
                micro[3][index_0][index_1][index_2],
            )
            index_3 = (
                np.searchsorted(macro[3][index_0][index_1][index_2], sampled_3) - 1
            )

            # Fifth dimension
            sampled_4 = np.interp(
                np.random.rand(),
                cumul[4][index_0][index_1][index_2][index_3],
                micro[4][index_0][index_1][index_2][index_3],
            )
            index_4 = (
                np.searchsorted(macro[4][index_0][index_1][index_2][index_3], sampled_4)
                - 1
            )

            # Sixth dimension
            sampled_5 = np.interp(
                np.random.rand(),
                cumul[5][index_0][index_1][index_2][index_3][index_4],
                micro[5][index_0][index_1][index_2][index_3][index_4],
            )

            # Append the sampled values
            sampled_values.append(
                [sampled_0, sampled_1, sampled_2, sampled_3, sampled_4, sampled_5]
            )

        return pd.DataFrame(sampled_values, columns=columns)

    if len(columns) == 5:
        part_1 = []
        part_2 = []
        part_3 = []
        part_4 = []
        part_5 = []
        part_6 = []
        for i in range(N):
            if i % 200000 == 0:
                print(i)

            # start_time = time.time()
            # First dimension
            sampled_0 = np.interp(np.random.rand(), cumul[0], micro[0])
            index_0 = np.searchsorted(macro[0], sampled_0) - 1
            end_time = time.time()
            # part_1.append(end_time - start_time)

            # start_time = time.time()
            # Second dimension
            sampled_1 = np.interp(
                np.random.rand(), cumul[1][index_0], micro[1][index_0]
            )
            index_1 = np.searchsorted(macro[1][index_0], sampled_1) - 1
            # end_time = time.time()
            # part_2.append(end_time - start_time)

            # start_time = time.time()
            # Third dimension
            sampled_2 = np.interp(
                np.random.rand(), cumul[2][index_0][index_1], micro[2][index_0][index_1]
            )
            index_2 = np.searchsorted(macro[2][index_0][index_1], sampled_2) - 1
            # end_time = time.time()
            # part_3.append(end_time - start_time)

            # start_time = time.time()
            # Fourth dimension
            sampled_3 = np.interp(
                np.random.rand(),
                cumul[3][index_0][index_1][index_2],
                micro[3][index_0][index_1][index_2],
            )
            # if sampled_3 > 1:
            #     sampled_3 = 1
            index_3 = (
                np.searchsorted(macro[3][index_0][index_1][index_2], sampled_3) - 1
            )
            # end_time = time.time()
            # part_4.append(end_time - start_time)

            # start_time = time.time()
            # Fifth dimension
            sampled_4 = np.interp(
                np.random.rand(),
                cumul[4][index_0][index_1][index_2][index_3],
                micro[4][index_0][index_1][index_2][index_3],
            )
            # end_time = time.time()
            # part_5.append(end_time - start_time)

            # start_time = time.time()
            # Append the sampled values
            sampled_values.append(
                [sampled_0, sampled_1, sampled_2, sampled_3, sampled_4]
            )
            # end_time = time.time()
            # part_6.append(end_time - start_time)

        # print("Part 1: ", np.mean(part_1))
        # print("Part 2: ", np.mean(part_2))
        # print("Part 3: ", np.mean(part_3))
        # print("Part 4: ", np.mean(part_4))
        # print("Part 5: ", np.mean(part_5))
        # print("Part 6: ", np.mean(part_6))

        return pd.DataFrame(sampled_values, columns=columns)

    # General case
    for _ in range(N):
        sample = []
        index = []

        for dim in range(len(columns)):
            cumul_aux = index_management(cumul[dim], index)
            micro_aux = index_management(micro[dim], index)

            # Sample the value for the current dimension
            sampled_value = np.interp(np.random.rand(), cumul_aux, micro_aux)
            sample.append(sampled_value)

            if (
                dim < len(columns) - 1
            ):  # Check if there are still more dimensions to process
                macro_aux = index_management(macro[dim], index)
                auxiliar = np.searchsorted(macro_aux, sampled_value) - 1
                index.append(auxiliar)

        sampled_values.append(sample)

    return pd.DataFrame(sampled_values, columns=columns)


def plot_correlated_variables(
    df: pd.DataFrame,
    columns: list,  # Por ahora todos tienen los mismos bines
    bin_size_diagonal: int = 100,
    bin_size_off_diagonal: int = 100,
    save: bool = False,
    plot: bool = True,
    density: bool = True,
    weight=True,
    filename: str = "correlated_histograms.png",
) -> None:

    fig, axes = plt.subplots(nrows=len(columns), ncols=len(columns), figsize=(15, 15))

    # Iterate through the rows and columns
    for i, col1 in enumerate(columns):
        for j, col2 in enumerate(columns):
            if i == j:
                # Diagonal plots: Plot histograms when col1 == col2
                if density:
                    df[col1].plot(
                        kind="hist",
                        ax=axes[i, j],
                        bins=bin_size_diagonal,
                        color="skyblue",
                        title=f"{col1}",
                        density=True,
                    )
                else:
                    df[col1].plot(
                        kind="hist",
                        ax=axes[i, j],
                        bins=bin_size_diagonal,
                        color="skyblue",
                        title=f"{col1}",
                    )
            else:
                # Off-diagonal plots: 2D histograms (heatmap-like) for col1 vs col2
                axes[i, j].hist2d(
                    df[col2], df[col1], bins=bin_size_off_diagonal, cmap="Blues"
                )
                axes[i, j].set_title(f"{col1} vs {col2}")
            # Set labels
            if i == len(columns) - 1:
                axes[i, j].set_xlabel(col2)
            if j == 0:
                axes[i, j].set_ylabel(col1)

    plt.tight_layout()  # Adjust layout to prevent overlap

    # Save the figure as a PNG file
    if save:
        plt.savefig(filename, dpi=300)

    if plot:
        plt.show()


def plot_results_barrido(
    df: pd.DataFrame,
    columns_order: list,
    micro_bins: list,
    macro_bins: list,
    type: str,
    name: str = "results_barrido",
) -> None:
    """ """

    # Create a 2x2 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    # Plot 1: Time for histograms
    axes[0, 0].plot(df.iloc[:, 0], df.iloc[:, 1], label=df.columns[1], marker="o")
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_xlabel("Numero de muestras")
    axes[0, 0].set_ylabel("Tiempo [s]")
    axes[0, 0].legend()
    axes[0, 0].grid()
    axes[0, 0].set_ylim(bottom=0)
    axes[0, 0].set_title(
        "Tiempo de calculo en funcion del numero de muestras", fontsize=10
    )

    # Plot 2: Time for sampling
    axes[0, 1].loglog(df.iloc[:, 0], df.iloc[:, 2], label=df.columns[2], marker="o")
    axes[0, 1].loglog(df.iloc[:, 0], df.iloc[:, 3], label=df.columns[3], marker="o")
    axes[0, 1].set_xlabel("Numero de muestras")
    axes[0, 1].set_ylabel("Tiempo [s]")
    axes[0, 1].legend()
    axes[0, 1].grid()
    axes[0, 0].set_ylim(bottom=1)
    axes[0, 1].set_title(
        "Tiempo de muestreo y corridaen funcion del numero de muestras", fontsize=10
    )

    # Plot 3: KL divergence_1d
    for i in range(4, 4 + len(columns_order)):
        axes[1, 0].loglog(df.iloc[:, 0], df.iloc[:, i], label=df.columns[i], marker="o")
    axes[1, 0].set_xlabel("Numero de muestras")
    axes[1, 0].set_ylabel("KL divergence 1D")
    axes[1, 0].legend()
    axes[1, 0].grid()
    axes[1, 0].set_title(
        "Divergencia KL 1D en funcion del numero de muestras", fontsize=10
    )

    # Plot 4: KL divergence_2d
    for i in range(
        4 + len(columns_order),
        4
        + len(columns_order)
        + int((len(columns_order) - 1) * len(columns_order) * 0.5),
    ):
        axes[1, 1].loglog(df.iloc[:, 0], df.iloc[:, i], label=df.columns[i], marker="o")
    axes[1, 1].set_xlabel("Numero de muestras")
    axes[1, 1].set_ylabel("KL divergence 2D")
    axes[1, 1].legend(fontsize="small")
    axes[1, 1].grid()
    axes[1, 1].set_title(
        "Divergencia KL 2D en funcion del numero de muestras", fontsize=10
    )

    # Add title with information of the run
    fig.suptitle(
        "Columns order: "
        + str(columns_order)
        + "\n"
        + "Micro bins: "
        + str(micro_bins)
        + "\n"
        + "Macro bins: "
        + str(macro_bins)
        + "\n"
        + "Type: "
        + type
    )

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the figure as a PNG file
    plt.savefig(name, dpi=300)

    # Close the figure to free up memory
    plt.close(fig)


def manage_n(
    amount: int,
    max_batch_size: float = 1e7,
    min_batch_size: float = 1e3,  # use base 10 for batch_size
) -> np.ndarray:
    """
    Calculate the number of samples to generate for each batch.

    Parameters
    ----------
    amount : int
        Total number of samples to generate.
    max_batch_size : float, optional
        Maximum number of samples to generate in a batch. The default is 1e7.
    min_batch_size : float, optional
        Minimum number of samples to generate in a batch. The default is 1e3.

    Returns
    -------
    n : np.ndarray[int]
        Array with the number of samples to generate for each batch.
    """
    if amount < min_batch_size:
        return np.array([amount]).astype(int)
    excess = amount / max_batch_size

    n = np.logspace(
        np.log10(min_batch_size),
        np.log10(max_batch_size),
        1 + int(np.log10(max_batch_size / min_batch_size)) * 2,
    )

    n = n.astype(int)

    if excess >= 1:
        n[1:] = n[1:] - n[:-1]
        # print((excess-int(excess))*max_batch_size)
        # print(0.00001 * max_batch_size)
        n = np.append(n, [max_batch_size] * (int(excess) - 1))
        if excess - int(excess) > 0:
            n = np.append(n, round((excess - int(excess)) * max_batch_size, 0))
        n = n.astype(int)
    else:
        n = n[n <= amount]
        if n[-1] != amount:
            n = np.append(n, amount)
        n[1:] = n[1:] - n[:-1]
        n = n.astype(int)

    return n


def save_information(
    type: list, N_max: list, columns_order: list, micro_bins: list, macro_bins: list
) -> None:
    """
    Save the information of the tests in a CSV file.

    Parameters
    ----------
    type : list
        List with the type of binning used in the tests.
    N_max : list
        List with the maximum number of samples used in the tests.
    columns_order : list
        List with the names of the columns used in the tests.
    micro_bins : list
        List with the number of micro bins used in the tests.
    macro_bins : list
        List with the number of macro bins used in the tests.
    """
    data = {
        "type": type,
        "N_max": N_max,
        "columns_order": columns_order,
        "micro_bins": micro_bins,
        "macro_bins": macro_bins,
    }
    df = pd.DataFrame(data)

    # Set the index to start from 1
    df.index = df.index + 1

    df.to_csv("index_information.csv", index=True)


def barrido_combinations(
    columns_order: list,
    micro_bins: list,
    macro_bins: list,
    N_max: list,
    type: list,
    min_batch_size=1e3,
    max_batch_size=1e7,
):
    columns_order_aux = []
    micro_bins_aux = []
    macro_bins_aux = []
    N_max_aux = []
    type_aux = []

    for columns_order_ in columns_order:
        for micro_bins_ in micro_bins:
            for macro_bins_ in macro_bins:
                for N_max_ in N_max:
                    for type_ in type:
                        columns_order_aux.append(columns_order_)
                        micro_bins_aux.append(micro_bins_)
                        macro_bins_aux.append(macro_bins_)
                        N_max_aux.append(
                            manage_n(
                                N_max_,
                                max_batch_size=max_batch_size,
                                min_batch_size=min_batch_size,
                            )
                        )
                        type_aux.append(type_)
    return columns_order_aux, micro_bins_aux, macro_bins_aux, N_max_aux, type_aux


def comparacion_barrido(
    index_max: int, rows_to_plot: list = None, result_name: str = "comparacion.png"
) -> None:

    if rows_to_plot is None:
        data = pd.read_csv(f"1/min_max.csv")
        rows_to_plot = [i for i in range(len(data))]
        del data

    # Create a figure
    fig, axes = plt.subplots(len(rows_to_plot), 2, figsize=(20, 3 * len(rows_to_plot)))

    for row in rows_to_plot:
        min_1d, max_1d, min_2d, max_2d = [], [], [], []
        for index in range(1, index_max + 1):
            data = pd.read_csv(f"{index}/min_max.csv")
            min_1d.append(data["min_1d"].iloc[row])
            max_1d.append(data["max_1d"].iloc[row])
            min_2d.append(data["min_2d"].iloc[row])
            max_2d.append(data["max_2d"].iloc[row])

        min_1d = np.array(min_1d)
        max_1d = np.array(max_1d)
        min_2d = np.array(min_2d)
        max_2d = np.array(max_2d)

        calification_1d = calificator(min_1d, max_1d)
        calification_2d = calificator(min_2d, max_2d)

        calification = calification_1d + calification_2d

        max_indexes = get_max_indexes(calification, 5)

        for index in range(1, index_max + 1):
            data = pd.read_csv(f"{index}/min_max.csv")
            axes[row, 0].plot(
                [index, index],
                [data["min_1d"].iloc[row], data["max_1d"].iloc[row]],
                # color="blue" if index % 2 == 0 else "red",
                color=(
                    "darkgreen"
                    if index - 1 == max_indexes[0]
                    else (
                        "green"
                        if index - 1 == max_indexes[1]
                        else (
                            "limegreen"
                            if index - 1 == max_indexes[2]
                            else (
                                "lightgreen"
                                if index - 1 == max_indexes[3]
                                else (
                                    "palegreen"
                                    if index - 1 == max_indexes[4]
                                    else "blue" if index % 2 == 0 else "red"
                                )
                            )
                        )
                    )
                ),
                # color = "darkgreen" if index == 1 else "green" if index == 2 else "limegreen" if index == 3 else "lightgreen" if index == 4 else "palegreen" if index == 5 else "blue" if index % 2 == 0 else "red",
                lw=2,
                marker="o",
            )  # Plot the box from min to max

            axes[row, 1].plot(
                [index, index],
                [data["min_2d"].iloc[row], data["max_2d"].iloc[row]],
                # color="blue" if index % 2 == 0 else "red",
                color=(
                    "darkgreen"
                    if index - 1 == max_indexes[0]
                    else (
                        "green"
                        if index - 1 == max_indexes[1]
                        else (
                            "limegreen"
                            if index - 1 == max_indexes[2]
                            else (
                                "lightgreen"
                                if index - 1 == max_indexes[3]
                                else (
                                    "palegreen"
                                    if index - 1 == max_indexes[4]
                                    else "blue" if index % 2 == 0 else "red"
                                )
                            )
                        )
                    )
                ),
                lw=2,
                marker="o",
            )  # Plot the box from min to max

        axes[row, 0].set_xticks(range(1, index_max + 1))
        axes[row, 0].set_xticklabels(range(1, index_max + 1), rotation=90)
        axes[row, 0].grid()
        axes[row, 0].set_yscale("log")
        axes[row, 0].set_title(
            "Divergencia_kl_1d (Min to Max), N = " + str(int(data["N"].iloc[row]))
        )
        axes[row, 0].set_xlabel("Index")

        axes[row, 1].set_xticks(range(1, index_max + 1))
        axes[row, 1].set_xticklabels(range(1, index_max + 1), rotation=90)
        axes[row, 1].grid()
        axes[row, 1].set_yscale("log")
        axes[row, 1].set_title(
            "Divergencia_kl_2d (Min to Max), N = " + str(int(data["N"].iloc[row]))
        )
        axes[row, 1].set_xlabel("Index")

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the figure as a PNG file
    plt.savefig(result_name, dpi=300)

    # Close the figure to free up memory
    plt.close(fig)


def get_min_max(
    index_max: int,
    amount_columns: int,
) -> None:

    original_path = os.getcwd()

    for i in range(1, index_max + 1):
        os.chdir(f"{original_path}/{i}")
        data = pd.read_csv(f"results.csv")
        kl_1d = data.iloc[:, 4 : 4 + amount_columns]
        kl_2d = data.iloc[:, 4 + amount_columns :]
        df_min_max = pd.DataFrame(columns=["N", "min_1d", "max_1d", "min_2d", "max_2d"])

        for j in range(len(kl_1d)):
            df_min_max.loc[len(df_min_max)] = [
                data.iloc[j, 0],
                kl_1d.iloc[j].min(),
                kl_1d.iloc[j].max(),
                kl_2d.iloc[j].min(),
                kl_2d.iloc[j].max(),
            ]

        df_min_max.to_csv("min_max.csv", index=False)

        os.chdir(original_path)


def calificator(min: np.ndarray, max: np.ndarray) -> np.ndarray:
    min_sorted = np.sort(min)
    # print(min)
    max_sorted = np.sort(max)
    # print(max)
    med = (min + max) / 2
    med_sorted = np.sort(med)
    # print(med)

    width = max - min

    width_sorted = np.sort(width)

    calification = np.zeros(len(min))
    # print("med ", c)
    # print(med)
    for i, (min, max, med, width) in enumerate(zip(min, max, med, width)):
        # Find the index of x in min_sorted, y in max_sorted, and z in med_sorted
        min_index = np.where(min_sorted == min)[0][0]  # Get the first occurrence
        max_index = np.where(max_sorted == max)[0][0]
        med_index = np.where(med_sorted == med)[0][0]
        width_index = np.where(width_sorted == width)[0][0]
        # print(z_index)

        # Add these indices to calification
        calification[i] = -(min_index + 1.3 * max_index + med_index + 0.5 * width_index)
        # calification[i] = -(min_index)

    return calification


def get_max_indexes(a: np.array, n: int) -> np.array:
    return a.argsort()[-n:][::-1]


def get_bin_edges(
    df: pd.DataFrame,
    columns_order: list,
    bin_size: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ """
    # Calculate edges 1d
    edges_1d = [
        edges
        for edges in [
            np.histogram(df[column], bins=bin_size, weights=df["wgt"])[1]
            for column in columns_order
        ]
    ]

    edges_1d = np.array(edges_1d)

    # Calculate edges 2d
    edges_2d_1, edges_2d_2 = zip(
        *[
            np.histogram2d(
                df[columns_order[i]],
                df[columns_order[j]],
                bins=bin_size,
                weights=df["wgt"],
            )[1:]
            for i in range(len(columns_order) - 1)
            for j in range(i + 1, len(columns_order))
        ]
    )

    edges_2d_1, edges_2d_2 = (
        np.array(edges_2d_1),
        np.array(edges_2d_2),
    )

    return edges_1d, edges_2d_1, edges_2d_2


def get_counts(
    df: pd.DataFrame,
    columns_order: list,
    edges_1d: list,
    edges_2d_1: list,
    edges_2d_2: list,
) -> tuple[np.ndarray, np.ndarray]:
    """ """

    counts_1d = np.array(
        [
            np.histogram(
                df[column], bins=edge, weights=df["wgt"] if "wgt" in df else None
            )[0]
            for column, edge in zip(columns_order, edges_1d)
        ]
    )

    counts_2d = np.array(
        [
            np.histogram2d(
                df[columns_order[i]],
                df[columns_order[j]],
                bins=[edges1, edges2],
                weights=df["wgt"] if "wgt" in df else None,
            )[0]
            for (i, j), edges1, edges2 in zip(
                [
                    (i, j)
                    for i in range(len(columns_order) - 1)
                    for j in range(i + 1, len(columns_order))
                ],
                edges_2d_1,
                edges_2d_2,
            )
        ]
    )

    # for (i, j), edges1, edges2 in zip(
    #     [
    #         (i, j)
    #         for i in range(len(columns_order) - 1)
    #         for j in range(i + 1, len(columns_order))
    #     ],
    #     edges_2d_1,
    #     edges_2d_2,
    # ):
    #     histo = np.histogram2d(
    #             df[columns_order[i]], df[columns_order[j]], bins=[edges1, edges2])[0]

    #     # histo = np.histogram2d(
    #     #         df[columns_order[i]], df[columns_order[j]], bins=[edges1, edges2], weights=df["wgt"] if "wgt" in df else None
    #     #     )[0]

    #     print('x')

    # counts_2d = np.array(
    #     [
    #         np.histogram2d(
    #             df[columns_order[i]], df[columns_order[j]], bins=[edges1, edges2], weights=df["wgt"] if "wgt" in df else None
    #         )[0]
    #         for (i, j), edges1, edges2 in zip(
    #             [
    #                 (i, j)
    #                 for i in range(len(columns_order) - 1)
    #                 for j in range(i + 1, len(columns_order))
    #             ],
    #             edges_2d_1,
    #             edges_2d_2,
    #         )
    #     ]
    # )

    return counts_1d, counts_2d


def get_time_and_counts(
    columns: list,
    cumul: list,
    micro: list,
    macro: list,
    N: int,
    edges_1d: list,
    edges_2d_1: list,
    edges_2d_2: list,
) -> tuple:
    """
    Calculate the time needed to calculate the cumulative histograms and to sample the data.
    Also calculate the counts of the sampled data for each column.

    Parameter:
    - df: DataFrame with the data.
    - columns: List with the names of the columns.
    - micro_bins: List with the number of bins for the cumulative histogram of the micro column.
    - macro_bins: List with the number of bins for the cumulative histogram of the macro column.
    - N: Number of samples to generate.
    - edges: List with the edges of the bins for each column to calculate the counts based on the original histogram.
    - type: Type of binning to use.

    Return:
    - Tuple with the time needed to calculate the cumulative histograms and to sample the data, and the counts of the sampled data for each column.
    """

    start_time = time.perf_counter()
    df_sampled = sample(cumul, micro, macro, columns, N)
    end_time = time.perf_counter()
    time_sampling = end_time - start_time

    counts_1d, counts_2d = get_counts(
        columns, df_sampled, edges_1d, edges_2d_1, edges_2d_2
    )

    return time_sampling, counts_1d, counts_2d


def plot_correlated_variables_counts(
    counts_1d: np.ndarray,
    counts_2d: np.ndarray,
    edges_1d: list,
    edges_2d_1: list,
    edges_2d_2: list,
    columns_order: list,
    filename: str = "correlated_histograms.png",
    save: bool = True,
    plot: bool = False,
) -> None:

    fig, axes = plt.subplots(
        nrows=len(columns_order), ncols=len(columns_order), figsize=(15, 15)
    )

    # Iterate through the rows and columns
    iterator = 0
    for i, col1 in enumerate(columns_order):
        for j, col2 in enumerate(columns_order):
            if i == j:
                # Diagonal plots: Plot histograms when col1 == col2
                axes[i, j].stairs(counts_1d[i], edges_1d[i], fill=True, color="skyblue")
                axes[i, j].set_title(f"{col1}")
                axes[i, j].grid()
                axes[i,j].set_yscale('log')
            if j > i:
                axes[i, j].pcolormesh(
                    edges_2d_2[iterator],
                    edges_2d_1[iterator],
                    counts_2d[iterator],
                    cmap="Blues",
                )
                axes[i, j].set_title(f"{col1} vs {col2}")
                axes[i, j].margins(x=0.4, y=0.4)
                axes[j, i].pcolormesh(
                    edges_2d_1[iterator],
                    edges_2d_2[iterator],
                    counts_2d[iterator].T,
                    cmap="Blues",
                )
                axes[j, i].set_title(f"{col2} vs {col1}")
                # Manually set the limits to add margins
                x_min, x_max = edges_2d_1[iterator][0], edges_2d_1[iterator][-1]
                y_min, y_max = edges_2d_2[iterator][0], edges_2d_2[iterator][-1]

                x_margin = (x_max - x_min) * 0.1  # 10% margin
                y_margin = (y_max - y_min) * 0.1  # 10% margin

                axes[j, i].set_xlim(x_min - x_margin, x_max + x_margin)
                axes[j, i].set_ylim(y_min - y_margin, y_max + y_margin)
                axes[j, i].set_yscale('log')
                iterator += 1

            # Set labels
            if i == len(columns_order) - 1:
                axes[i, j].set_xlabel(col2)
            if j == 0:
                axes[i, j].set_ylabel(col1)

    plt.tight_layout()  # Adjust layout to prevent overlap

    if plot:
        plt.show()

    if save:
        # Save the figure as a PNG file
        plt.savefig(filename, dpi=300)

    # Close the figure to free up memory
    plt.close(fig)


def barrido(
    columns_order: list,
    micro_bins: list,
    macro_bins: list,
    N: list,
    type: list,
    df: pd.DataFrame,
    bins_for_comparation: int = 100,
    value_to_replace_0: float = 1e-6,
) -> None:
    """ """

    for index, columns_order, micro_bins, macro_bins, N, type in zip(
        range(1, len(columns_order) + 1),
        columns_order,
        micro_bins,
        macro_bins,
        N,
        type,
    ):
        print(f"Index: {index}")

        # Create a folder to save results
        if not os.path.exists(f"{index}"):
            os.makedirs(f"{index}")

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
            filename=f"{index}/original.png",
        )

        # Calculate the histograms for the original data
        start_time = time.perf_counter()
        cumul, micro, macro = calculate_cumul_micro_macro(
            df, columns_order, micro_bins, macro_bins, type=type
        )
        time_histo = time.perf_counter() - start_time

        # Initialize dataframe to store the results of the index
        df_results = pd.DataFrame(
            columns=[
                "N",
                "Tiempo de calculo de los histogramas",
                "Tiempo de muestreo",
                "Tiempo de corrida",
            ]
            + columns_order
            + [
                columns_order[i] + " - " + columns_order[j]
                for i in range(len(columns_order) - 1)
                for j in range(i + 1, len(columns_order))
            ]
        )

        # Initialize variables to store the results of the batches
        (
            time_sample_container,
            time_running_container,
            counts_1d_container,
            counts_2d_container,
        ) = (0, 0, np.zeros_like(counts_1d_original), np.zeros_like(counts_2d_original))

        # Run the tests in batches
        for j, n in enumerate(N):
            print(f"Sampling {n} samples to get {sum(N[: j + 1])} samples in total")

            start_time = time.perf_counter()
            df_sampled = sample(cumul, micro, macro, columns_order, n)
            time_sample_container += time.perf_counter() - start_time

            (
                counts_1d_batch,
                counts_2d_batch,
            ) = get_counts(
                df_sampled,
                columns_order,
                edges_1d,
                edges_2d_1,
                edges_2d_2,
            )

            del df_sampled

            counts_1d_container += counts_1d_batch
            counts_2d_container += counts_2d_batch

            # normalize counts_container and take aways the zeros
            counts_1d_container_normalized, counts_2d_container_normalized = (
                normalize_counts(
                    counts_1d_container,
                    counts_2d_container,
                    edges_1d,
                    edges_2d_1,
                    edges_2d_2,
                    value_to_replace_0=value_to_replace_0,
                )
            )

            # Save plots of variables for each number of samples
            plot_correlated_variables_counts(
                counts_1d=counts_1d_container_normalized,
                counts_2d=counts_2d_container_normalized,
                edges_1d=edges_1d,
                edges_2d_1=edges_2d_1,
                edges_2d_2=edges_2d_2,
                columns_order=columns_order,
                filename=f"{index}/{sum(N[:j+1])}.png",
            )
            # print("synthethic different than zero: ",
            kl_1d = [
                abs(
                    np.sum(
                        rel_entr(original[synthetic != 0], synthetic[synthetic != 0])
                    )
                )
                / original[synthetic != 0].size
                for original, synthetic in zip(
                    counts_1d_original, counts_1d_container_normalized
                )
            ]
            kl_2d = [
                abs(
                    np.sum(
                        rel_entr(original[synthetic != 0], synthetic[synthetic != 0])
                    )
                )
                / original[synthetic != 0].size
                for original, synthetic in zip(
                    counts_2d_original, counts_2d_container_normalized
                )
            ]

            time_running_container += time.perf_counter() - start_time

            df_results.loc[len(df_results)] = (
                [
                    sum(N[: j + 1]),
                    time_histo,
                    time_sample_container,
                    time_running_container,
                ]
                + kl_1d
                + kl_2d
            )

        # Save the results in a .csv file
        df_results.to_csv(f"{index}/results.csv", index=False)

        plot_results_barrido(
            df_results,
            columns_order,
            micro_bins,
            macro_bins,
            type,
            name=f"{index}/results.png",
        )


def normalize_counts(
    counts_1d: np.ndarray,
    counts_2d: np.ndarray,
    edges_1d: np.ndarray,
    edges_2d_1: np.ndarray,
    edges_2d_2: np.ndarray,
    value_to_replace_0: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:

    delta_1 = edges_1d[0][1] - edges_1d[0][0]

    counts_1d = np.array([counts / counts.sum() / delta_1 for counts in counts_1d])
    if value_to_replace_0 is not None:
        counts_1d[counts_1d == 0] = value_to_replace_0

    delta_2_1 = edges_2d_1[0][1] - edges_2d_1[0][0]
    delta_2_2 = edges_2d_2[0][1] - edges_2d_2[0][0]

    counts_2d = np.array(
        [counts / counts.sum() / delta_2_1 / delta_2_2 for counts in counts_2d]
    )

    if value_to_replace_0 is not None:
        counts_2d[counts_2d == 0] = value_to_replace_0

    return counts_1d, counts_2d


def combine_images(num_folders: int, image_name: str = "1000000.png"):
    # Assuming all

    # Path to the main directory containing folders 1, 2, ..., 12
    main_dir = "./"  # Replace with the correct path if running from a different folder
    output_path = os.path.join(main_dir, "combined.png")  # Path for the output image

    # Parameters for the grid of images
    images_per_folder = 3
    images_names = [image_name, "original.png", "results.png"]

    # Assuming all images are the same size, get the dimensions of a sample image
    sample_image = Image.open(os.path.join(main_dir, "1", images_names[0]))
    img_width, img_height = sample_image.size
    sample_image.close()

    # Maximum pixels for the output image
    max_pixels = 3e7

    # Calculate reduction factor to keep final image size within max_pixels
    reducing_factor = np.sqrt(
        max_pixels / (img_width * img_height * images_per_folder * num_folders)
    )

    # New dimensions for each individual resized image
    new_img_width = int(img_width * reducing_factor)
    new_img_height = int(img_height * reducing_factor)

    # Define separator thickness and additional padding for labels
    row_separator_thickness = 5  # Thicker line between rows
    column_separator_thickness = 2  # Thinner line between columns
    label_padding = int(new_img_height * 0.1)  # Additional padding for label text

    # Dimensions of the final combined image, including padding for labels
    combined_width = (
        new_img_width * images_per_folder
        + (images_per_folder - 1) * column_separator_thickness
    )
    combined_height = (new_img_height + label_padding) * num_folders + (
        num_folders - 1
    ) * row_separator_thickness

    # Create a blank image with extra space for separators and labels
    combined_img = Image.new("RGB", (combined_width, combined_height), "white")

    # Load a small font
    font_size = int(new_img_height * 0.1)  # Relative font size based on image height
    try:

        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size
        )
    except IOError:
        print("prrr")
        font = ImageFont.load_default()  # Use default font if custom font fails

    # Draw object to add lines and text
    draw = ImageDraw.Draw(combined_img)

    # Loop through each folder and each image within it
    for i in range(1, num_folders + 1):
        folder_path = os.path.join(main_dir, str(i))

        # Calculate y position with row separator space
        y_offset = (i - 1) * (
            new_img_height + label_padding + row_separator_thickness
        ) - int(new_img_height * 0.1)

        # Add row label at the top of each row
        label_text = f"INDEX = {i}"
        text_width, text_height = draw.textsize(label_text, font=font)
        text_x = (combined_width - text_width) // 2
        draw.text(
            (text_x, y_offset + int(new_img_height * 0.1)),
            label_text,
            fill="black",
            font=font,
        )

        for j in range(1, images_per_folder + 1):
            img_path = os.path.join(folder_path, images_names[j - 1])
            img = Image.open(img_path)

            # Resize the image according to the reducing factor
            img_resized = img.resize((new_img_width, new_img_height))

            # Calculate x position with column separator space
            x_offset = (j - 1) * (new_img_width + column_separator_thickness)

            # Paste the resized image into the combined image
            combined_img.paste(
                img_resized, (x_offset, y_offset + text_height + label_padding)
            )
            img.close()

    # Draw separators
    for i in range(1, num_folders):
        # Horizontal (row) separators
        y_position = (
            i * (new_img_height + label_padding + row_separator_thickness)
            - row_separator_thickness // 2
        )
        draw.line(
            [(0, y_position), (combined_width, y_position)],
            fill="black",
            width=row_separator_thickness,
        )

    for j in range(1, images_per_folder):
        # Vertical (column) separators
        x_position = (
            j * (new_img_width + column_separator_thickness)
            - column_separator_thickness // 2
        )
        draw.line(
            [(x_position, 0), (x_position, combined_height)],
            fill="black",
            width=column_separator_thickness,
        )

    # Save the final combined image
    combined_img.save(output_path)

    # Free up resources
    combined_img.close()

