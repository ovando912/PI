import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import glob
import shutil
import openmc
import kdsource.histograms as kdh
import kdsource as kds
import subprocess


plt.rcParams.update(
    {
        "text.usetex": True,  # Usa LaTeX para el texto
        "font.family": "serif",  # Fuente tipo serif (similar a LaTeX)
        "font.serif": ["Computer Modern Roman"],  # Usa Computer Modern
        "axes.labelsize": 20,  # Tamaño de etiquetas de ejes
        "font.size": 20,  # Tamaño general de fuente
        "legend.fontsize": 18,  # Tamaño de fuente en la leyenda
        "xtick.labelsize": 18,  # Tamaño de fuente en los ticks de x
        "ytick.labelsize": 18,  # Tamaño de fuente en los ticks de y
        "font.weight": "bold",  # Fuente en negrita
        "axes.titleweight": "bold",  # Títulos de ejes en negrita
    }
)


class Simulation:
    def __init__(
        self,
        simulacion_numero,
        geometria,
        z0,
        z_track,
        fuente,
        z_for_spectral_tally,
        num_particles,
        WW,
        path,
        folder,
        statepoint_name,
        trackfile_name,
        columns_order,
        micro_bins,
        macro_bins,
        binning_type,
        user_defined_edges,
        output_XML_name,
        num_resampling,
        factor_normalizacion=1,
        factor_normalizacion_siguiente=1,
    ):
        self.simulacion_numero = simulacion_numero
        self.geometria = geometria
        self.z0 = z0
        self.z_track = z_track
        self.fuente = fuente
        self.z_for_spectral_tally = z_for_spectral_tally
        self.num_particles = num_particles
        self.WW = WW
        self.path = path
        self.folder = folder
        self.statepoint_name = statepoint_name
        self.trackfile_name = trackfile_name
        self.columns_order = columns_order
        self.micro_bins = micro_bins
        self.macro_bins = macro_bins
        self.binning_type = binning_type
        self.user_defined_edges = user_defined_edges
        self.output_XML_name = output_XML_name
        self.trackfile_resampled_name = self.trackfile_name.replace(".h5", "_resampled")
        self.num_resampling = num_resampling
        self.factor_normalizacion = factor_normalizacion
        self.factor_normalizacion_siguiente = factor_normalizacion_siguiente

    def run_simulation(self) -> None:
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

            z0 (float): Posición en z donde empieza la simulación.
            z_track (float): Posición en z para la superficie de track.
            N_particles (int): Número de partículas a simular en total.
            outfile_name (str): Nombre del archivo de salida. Si es None, se usa el nombre por defecto.
            WW (bool): Si se activa, se generan weight windows.
        """
        # ---------------------------------------------------------------------------------------------------------------------------------------
        # Configuración de secciones eficaces
        # ---------------------------------------------------------------------------------------------------------------------------------------
        openmc.config["cross_sections"] = (
            "/home/lucas/Documents/Proyecto_Integrador/endfb-viii.0-hdf5/cross_sections.xml"
        )

        # ---------------------------------------------------------------------------------------------------------------------------------------
        # Configuración de la carpeta de trabajo
        # ---------------------------------------------------------------------------------------------------------------------------------------
        # Si se especifica una carpeta, se crea y se cambia a esa carpeta. Si no, se usa la carpeta actual.
        if self.folder is not None:
            if not os.path.exists(self.folder):
                os.makedirs(self.folder)
            os.chdir(self.folder)

        # ----------------------------------------------------------------------------------------------------------------------------------------
        # Procesamiento de la fuente
        # ----------------------------------------------------------------------------------------------------------------------------------------
        # Si la longitud de la fuente es 1, se asume que es un archivo de fuente.
        # Si la longitud de la fuente es 2, se asume que es una fuente independiente.
        if len(self.fuente) == 1:
            source = openmc.FileSource(self.fuente[0])
        elif len(self.fuente) == 2:
            fuente_energia, fuente_direccion = self.fuente
            source = openmc.IndependentSource()
            source.particle = "neutron"

            # Distribución espacial: se coloca en la región central de la geometría
            L_x, L_y = self.geometria[1], self.geometria[2]
            x_dist = openmc.stats.Uniform(-L_x / 2, L_x / 2)
            y_dist = openmc.stats.Uniform(-L_y / 2, L_y / 2)
            z_dist = openmc.stats.Discrete(
                self.z0 + 1e-6, 1
            )  # Se fija z muy cerca de z=0
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

        # ---------------------------------------------------------------------------------------------------------------------------------------
        # Procesamiento de la geometría
        # ---------------------------------------------------------------------------------------------------------------------------------------
        # Extraer parámetros geométricos
        vacio = self.geometria[0]
        L_x, L_y, L_z = self.geometria[1:4]
        if vacio:
            L_x_vacio, L_y_vacio = self.geometria[4:6]

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
            "z_min": openmc.ZPlane(z0=self.z0, boundary_type="vacuum"),
            "z_max": openmc.ZPlane(z0=L_z, boundary_type="vacuum"),
        }

        # Se agrega la superficie de registro para generar el track file
        if self.z_track is not None:
            surfaces.update(
                {
                    "z_track": openmc.ZPlane(
                        z0=self.z_track, boundary_type="transmission", surface_id=70
                    )
                }
            )

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

        # Para fuente tipo FileSource se traduce la superficie inferior para posicionar z0.
        # Sino se hace entonces las particulas aparecer fuera de la geometria.
        if len(self.fuente) == 1:
            surfaces["z_min"].translate(vector=(0, 0, -1e-6), inplace=True)

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
            if self.z_track is not None:
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
                        region=region_vacio & -surfaces["z_track"],
                        fill=None,
                        name="vacio1",
                    )
                )
                universe.add_cell(
                    openmc.Cell(
                        region=region_vacio & +surfaces["z_track"],
                        fill=None,
                        name="vacio2",
                    )
                )
            else:
                universe.add_cell(
                    openmc.Cell(
                        region=region_externa & ~region_vacio,
                        fill=mat_agua,
                        name="agua",
                    )
                )
                universe.add_cell(
                    openmc.Cell(region=region_vacio, fill=None, name="vacio")
                )
        else:
            if self.z_track is not None:
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

        # ---------------------------------------------------------------------------------------------------------------------------------------
        # Configuración de settings
        # ---------------------------------------------------------------------------------------------------------------------------------------
        settings = openmc.Settings()
        settings.surf_source_write = {"surface_ids": [70], "max_particles": 20000000}
        settings.run_mode = "fixed source"
        settings.batches = 100
        settings.particles = int(self.num_particles / 100)
        settings.source = source

        # Se definen las ventanas de peso
        if self.WW:
            # Define weight window spatial mesh
            ww_mesh = openmc.RegularMesh()
            ww_mesh.dimension = (10, 10, 10)
            ww_mesh.lower_left = (-L_x / 2, -L_y / 2, self.z0)
            ww_mesh.upper_right = (L_x / 2, L_y / 2, L_z)

            # Create weight window object and adjust parameters
            wwg = openmc.WeightWindowGenerator(
                method="magic",
                mesh=ww_mesh,
                max_realizations=settings.batches,
                energy_bounds=[0, 1, 1e3, 1e7],
                update_interval=2,
            )

            # Add generator to Settings instance
            settings.weight_window_generators = wwg
        settings.export_to_xml()

        # ---------------------------------------------------------------------------------------------------------------------------------------
        # Configuración de tallies
        # ---------------------------------------------------------------------------------------------------------------------------------------
        # Tally: malla para flujo total
        mesh = openmc.RectilinearMesh()
        mesh.x_grid = np.linspace(-L_x / 2, L_x / 2, 2)
        mesh.y_grid = np.linspace(-L_y / 2, L_y / 2, 2)
        mesh.z_grid = np.linspace(self.z0, L_z, 51)
        mesh_filter = openmc.MeshFilter(mesh)
        tally_flux_total = openmc.Tally(name="flux_total")
        tally_flux_total.filters = [mesh_filter]
        tally_flux_total.scores = ["flux"]

        # Tally: malla para flujo en vacio
        if vacio:
            mesh_vacio = openmc.RectilinearMesh()
            mesh_vacio.x_grid = np.linspace(-L_x_vacio / 2, L_x_vacio / 2, 2)
            mesh_vacio.y_grid = np.linspace(-L_y_vacio / 2, L_y_vacio / 2, 2)
            mesh_vacio.z_grid = np.linspace(self.z0, L_z, 51)
            mesh_filter_vacio = openmc.MeshFilter(mesh_vacio)
            tally_flux_vacio = openmc.Tally(name="flux_vacio")
            tally_flux_vacio.filters = [mesh_filter_vacio]
            tally_flux_vacio.scores = ["flux"]

        tallies = openmc.Tallies(
            [tally_flux_total, tally_flux_vacio] if vacio else [tally_flux_total]
        )

        # Tally: superficie para espectro en 1m
        def make_spectrum_tally(L_x, L_y, L_z, name):
            mesh_total = openmc.RectilinearMesh()
            mesh_total.x_grid = np.linspace(-L_x / 2, L_x / 2, 2)
            mesh_total.y_grid = np.linspace(-L_y / 2, L_y / 2, 2)
            mesh_total.z_grid = np.linspace(L_z - 1, L_z, 2)

            tally_surface = openmc.Tally(name=name)
            tally_surface.filters = [
                openmc.MeshFilter(mesh_total),
                openmc.EnergyFilter(np.logspace(-3, 7, 50)),
            ]
            tally_surface.scores = ["flux"]
            return tally_surface

        for Z in self.z_for_spectral_tally:
            if self.z0 <= Z <= L_z:
                tallies.append(
                    make_spectrum_tally(L_x, L_y, Z, f"espectro_total_{Z}cm")
                )
                if vacio:
                    tallies.append(
                        make_spectrum_tally(
                            L_x_vacio, L_y_vacio, Z, f"espectro_vacio_{Z}cm"
                        )
                    )

        tallies.export_to_xml()

        # ---------------------------------------------------------------------------------------------------------------------------------------
        # Limpieza de archivos previos y ejecución de la simulación
        # ---------------------------------------------------------------------------------------------------------------------------------------
        for file in glob.glob("statepoint.*.h5"):
            os.remove(file)
        if os.path.exists("summary.h5"):
            os.remove("summary.h5")

        openmc.run()

        # ---------------------------------------------------------------------------------------------------------------------------------------
        # Procesamiento de los resultados
        # ---------------------------------------------------------------------------------------------------------------------------------------
        # Mover archivos de salida según tipo de fuente
        statepoint_files = glob.glob("statepoint.*.h5")

        for file in statepoint_files:
            shutil.move(file, self.statepoint_name)

        if self.z_track is not None and self.trackfile_name is not None:
            shutil.move("surface_source.h5", self.trackfile_name)

        # ----------------------------------------------------------------------------------------------------------------------------------------
        # Return to the previous directory if a folder was specified
        # ----------------------------------------------------------------------------------------------------------------------------------------
        if self.folder is not None:
            os.chdir("..")

    def plot_flux(self, save=False):

        def tally_flux_2_df(path, tally_name, factor_normalizacion):
            sp = openmc.StatePoint(path)
            tally = sp.get_tally(name=tally_name)
            df = tally.get_pandas_dataframe()
            if len(df.columns) == 7:
                df.columns = [
                    "x",
                    "y",
                    "z",
                    "nuclide",
                    "score",
                    "mean",
                    "std.dev.",
                ]
            else:
                df.columns = [
                    "x",
                    "y",
                    "z",
                    "nuclide",
                    "score",
                    "mean",
                ]
                df["std.dev."] = 1000

            volumen_celda = (
                tally.filters[0]._mesh.total_volume
                / tally.filters[0]._mesh.num_mesh_cells
            )
            df["mean_norm"] = df["mean"] / volumen_celda * factor_normalizacion
            df["std.dev._norm"] = df["std.dev."] / volumen_celda
            z_min = tally.filters[0]._mesh.lower_left[2]
            z_max = tally.filters[0]._mesh.upper_right[2]
            z = np.linspace(z_min, z_max, tally.filters[0]._mesh.num_mesh_cells + 1)
            z_midpoints = (z[:-1] + z[1:]) / 2
            df["z_midpoints"] = z_midpoints
            return df

        def get_tally_agua(df_total, df_vacio):
            volumen_total = df_total["mean"].sum() / df_total["mean_norm"].sum()
            volumen_vacio = df_vacio["mean"].sum() / df_vacio["mean_norm"].sum()
            volumen_agua = volumen_total - volumen_vacio

            df_agua = df_total.copy()
            df_agua["mean"] = df_agua["mean"] - df_vacio["mean"]
            df_agua["std.dev."] = np.sqrt(
                df_agua["std.dev."] ** 2 + df_vacio["std.dev."] ** 2
            )  # Revisar esto si hace falta. No se reviso hasta el momento.
            df_agua["mean_norm"] = df_agua["mean"] / volumen_agua
            df_agua["std.dev._norm"] = df_agua["std.dev."] / volumen_agua

            return df_agua

        df_flux_total = tally_flux_2_df(
            self.folder + self.statepoint_name,
            tally_name="flux_total",
            factor_normalizacion=self.factor_normalizacion,
        )
        df_flux_vacio = tally_flux_2_df(
            self.folder + self.statepoint_name,
            tally_name="flux_vacio",
            factor_normalizacion=self.factor_normalizacion,
        )
        df_flux_agua = get_tally_agua(df_flux_total, df_flux_vacio)

        plt.figure(figsize=(10, 6))
        plt.plot(
            df_flux_agua["z_midpoints"],
            df_flux_agua["mean_norm"],
            label="Flujo en agua",
            color="blue",
        )
        plt.plot(
            df_flux_vacio["z_midpoints"],
            df_flux_vacio["mean_norm"],
            label="Flujo en vacío",
            color="red",
        )
        plt.plot(
            df_flux_total["z_midpoints"],
            df_flux_total["mean_norm"],
            label="Flujo total",
            color="green",
        )
        plt.xlabel("z [cm]")
        plt.ylabel("Flujo [cm$^{-2}$ s$^{-1}$]")
        plt.title("Flujo en funcion de z")
        plt.legend()
        plt.grid()
        plt.yscale("log")
        if save:
            plt.savefig(
                self.folder + f"flujo_{self.simulacion_numero}.png",
                bbox_inches="tight",
                dpi=300,
            )
        plt.show()

    def plot_espectro(self, z=None, save=False):
        def tally_espectro_2_df(path, tally_name, factor_normalizacion):
            sp = openmc.StatePoint(path)
            tally = sp.get_tally(name=tally_name)
            df = tally.get_pandas_dataframe()
            df.columns = [
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
            volumen_celda = (
                tally.filters[0]._mesh.total_volume
                / tally.filters[0]._mesh.num_mesh_cells
            )
            df["mean_norm"] = df["mean"] / volumen_celda * factor_normalizacion
            df["std.dev._norm"] = df["std.dev."] / volumen_celda
            df["E_mid"] = (df["E_min"] + df["E_max"]) / 2
            return df

        if z is None:
            for Z in self.z_for_spectral_tally:
                df_espectro_total = tally_espectro_2_df(
                    self.folder + self.statepoint_name,
                    tally_name=f"espectro_total_{Z}cm",
                    factor_normalizacion=self.factor_normalizacion,
                )
                df_espectro_vacio = tally_espectro_2_df(
                    self.folder + self.statepoint_name,
                    tally_name=f"espectro_vacio_{Z}cm",
                    factor_normalizacion=self.factor_normalizacion,
                )
                df_espectro_agua = get_tally_agua(df_espectro_total, df_espectro_vacio)

                plt.figure(figsize=(10, 6))
                plt.plot(
                    df_espectro_agua["E_mid"],
                    df_espectro_agua["mean_norm"],
                    label=f"Espectro en agua {Z} cm",
                    color="blue",
                )
                plt.plot(
                    df_espectro_total["E_mid"],
                    df_espectro_total["mean_norm"],
                    label=f"Espectro total {Z} cm",
                    color="green",
                )
                plt.plot(
                    df_espectro_vacio["E_mid"],
                    df_espectro_vacio["mean_norm"],
                    label=f"Espectro en vacío {Z} cm",
                    color="red",
                )
                plt.xlabel("E [eV]")
                plt.ylabel("Flujo [cm$^{-2}$ s$^{-1}$ eV$^{-1}$]")
                plt.title(f"Espectro en {Z} cm")
                plt.legend()
                plt.grid()
                plt.yscale("log")
                plt.xscale("log")

                if save:
                    plt.savefig(
                        self.folder + f"espectro_{self.simulacion_numero}_{Z}.png",
                        bbox_inches="tight",
                        dpi=300,
                    )
                plt.show()
        else:
            df_espectro_total = tally_espectro_2_df(
                self.folder + self.statepoint_name,
                tally_name=f"espectro_total_{z}cm",
                factor_normalizacion=self.factor_normalizacion,
            )
            df_espectro_vacio = tally_espectro_2_df(
                self.folder + self.statepoint_name,
                tally_name=f"espectro_vacio_{z}cm",
                factor_normalizacion=self.factor_normalizacion,
            )
            df_espectro_agua = get_tally_agua(df_espectro_total, df_espectro_vacio)

            plt.figure(figsize=(10, 6))
            plt.plot(
                df_espectro_agua["E_mid"],
                df_espectro_agua["mean_norm"],
                label=f"Espectro en agua {z} cm",
                color="blue",
            )
            plt.plot(
                df_espectro_total["E_mid"],
                df_espectro_total["mean_norm"],
                label=f"Espectro total {z} cm",
                color="green",
            )
            plt.plot(
                df_espectro_vacio["E_mid"],
                df_espectro_vacio["mean_norm"],
                label=f"Espectro en vacío {z} cm",
                color="red",
            )
            plt.xlabel("E [eV]")
            plt.ylabel("Flujo [cm$^{-2}$ s$^{-1}$ eV$^{-1}$]")
            plt.title(f"Espectro en {z} cm")
            plt.legend()
            plt.grid()
            plt.yscale("log")
            plt.xscale("log")

            if save:
                plt.savefig(
                    self.folder + f"espectro_{self.simulacion_numero}_{z}.png",
                    bbox_inches="tight",
                    dpi=300,
                )
            plt.show()

    def generate_xml(self):
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        os.chdir(self.folder)

        print(self.trackfile_name)

        trackfile = kdh.SurfaceTrackProcessor(self.trackfile_name, self.num_particles)

        self.factor_normalizacion_siguiente = (
            trackfile.factor_normalizacion * self.factor_normalizacion
        )

        trackfile.configure_binning(
            columns=self.columns_order,
            micro_bins=self.micro_bins,
            macro_bins=self.macro_bins,
            user_defined_macro_edges=self.user_defined_edges,
        )

        trackfile.plot_correlated_variables(filename="original.png")

        trackfile.load_simulation_info(
            geometria=self.geometria,
            z0=self.z_track,  # No esta mal. Antes me referia con z0 a la superficie de registro, y en esta clase z0 representa
            # la posicion de la fuente. z_track cumple la funcion de indicar donde se ubico la superficie de registro.
            fuente_original=self.fuente,
        )

        print(
            f"""
        ╔══════════════════════════════════════╗
        ║       Partículas Registradas         ║
        ╠══════════════════════════════════════╣
        ║ Total         : {len(trackfile.df):.1e}              ║
        ║ mu = 1 (%)    : {trackfile.df.loc[trackfile.df['mu'] == 1].shape[0] / len(trackfile.df) * 100:.2f}%               ║
        ╚══════════════════════════════════════╝
        """
        )

        trackfile.save_to_xml(self.output_XML_name)

        # tree = trackfile.Tree

        os.chdir("..")

    def resample(self, plot=True, compare=True, bins2compare=100):
        output_path = os.path.join(
            self.path, self.folder, self.trackfile_resampled_name
        )
        source_path = os.path.join(self.path, self.folder, self.output_XML_name)
        command = [
            "kdtool",
            "resample",
            "-o",
            output_path,
            "-n",
            str(self.num_resampling),
            "-m",
            "2",
            source_path,
        ]
        # Ejecutar el comando usando subprocess
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error al ejecutar kdtool: {e}")
            return
        # !kdtool resample -o {output_path} -n {sampling.num_resampling} -m 2 {source_path}
        kds.SurfaceSourceFile(output_path + ".mcpl.gz").save_source_file(
            output_path + ".h5"
        )

        if plot:
            trackfile_resampled = kdh.SurfaceTrackProcessor(
                self.folder + self.trackfile_resampled_name + ".h5", self.num_particles
            )

            trackfile_resampled.configure_binning(
                columns=self.columns_order,
                micro_bins=self.micro_bins,
                macro_bins=self.macro_bins,
                user_defined_macro_edges=self.user_defined_edges,
            )

            trackfile_resampled.plot_correlated_variables(
                filename=self.folder + "resampled.png"
            )
            del trackfile_resampled

        if compare:
            # Cargo los archivos en DFs
            # Cargar el archivo original en un DataFrame de pandas
            df_original = kds.SurfaceSourceFile(
                self.folder + self.trackfile_name, domain={"w": [0, 2]}
            ).get_pandas_dataframe()[["x", "y", "ln(E0/E)", "mu", "phi", "wgt"]]
            hist_lethargy_original, edges_lethargy_original = np.histogram(
                df_original["ln(E0/E)"],
                bins=bins2compare,
                density=True,
                weights=df_original["wgt"],
            )
            hist_x_original, edges_x_original = np.histogram(
                df_original["x"],
                bins=bins2compare,
                density=True,
                weights=df_original["wgt"],
            )
            hist_y_original, edges_y_original = np.histogram(
                df_original["y"],
                bins=bins2compare,
                density=True,
                weights=df_original["wgt"],
            )
            hist_mu_original, edges_mu_original = np.histogram(
                df_original["mu"],
                bins=bins2compare,
                density=True,
                weights=df_original["wgt"],
            )
            hist_phi_original, edges_phi_original = np.histogram(
                df_original["phi"],
                bins=bins2compare,
                density=True,
                weights=df_original["wgt"],
            )
            del df_original

            # Cargar el archivo sintético en un DataFrame de pandas
            df_synthetic = kds.SurfaceSourceFile(
                self.folder + self.trackfile_resampled_name + ".h5",
                domain={"w": [0, 2]},
            ).get_pandas_dataframe()[["x", "y", "ln(E0/E)", "mu", "phi", "wgt"]]
            hist_lethargy_synthetic, _ = np.histogram(
                df_synthetic["ln(E0/E)"], bins=edges_lethargy_original, density=True
            )
            hist_x_synthetic, _ = np.histogram(
                df_synthetic["x"], bins=edges_x_original, density=True
            )
            hist_y_synthetic, _ = np.histogram(
                df_synthetic["y"], bins=edges_y_original, density=True
            )
            hist_mu_synthetic, _ = np.histogram(
                df_synthetic["mu"], bins=edges_mu_original, density=True
            )
            hist_phi_synthetic, _ = np.histogram(
                df_synthetic["phi"], bins=edges_phi_original, density=True
            )
            del df_synthetic

            def plot_comparison(
                hist_original, hist_synthetic, edges, variable, folder=None
            ):
                """
                Esta función genera una comparación gráfica entre dos histogramas (original y sintético) para una variable dada.

                Parámetros:
                - hist_original: numpy.ndarray
                    Histograma de la variable original.
                - hist_synthetic: numpy.ndarray
                    Histograma de la variable sintética.
                - edges: numpy.ndarray
                    Bordes de los bins del histograma.
                - variable: str
                    Nombre de la variable que se está comparando.
                - folder: str, opcional
                    Carpeta donde se guardará la figura generada. Por defecto es una cadena vacía, lo que significa que la figura se guardará en el directorio actual.

                La función genera una figura con dos subgráficos:
                1. El primer subgráfico muestra los histogramas original y sintético en escala logarítmica.
                2. El segundo subgráfico muestra el error relativo entre los histogramas original y sintético.

                La figura se guarda como un archivo PNG con un nombre basado en la variable comparada.
                """
                fig = plt.figure(figsize=(16 * 1.25, 9 * 0.8))
                gs = gridspec.GridSpec(
                    2, 1, height_ratios=[2.5, 1]
                )  # La primera fila es el doble de alta que la segunda

                axs = [
                    fig.add_subplot(gs[i]) for i in range(2)
                ]  # Crear una lista de ejes

                plt.subplots_adjust(
                    left=0.09,
                    right=0.97,
                    top=0.95,
                    bottom=0.05,
                    wspace=0.4,
                    hspace=0.25,
                )

                # Graficar hist_original y hist_synthetic en el primer gráfico de fig
                axs[0].plot(
                    (edges[:-1] + edges[1:]) / 2,
                    hist_original,
                    label="Original",
                    color="blue",
                    linestyle="--",
                )
                axs[0].plot(
                    (edges[:-1] + edges[1:]) / 2,
                    hist_synthetic,
                    label="Sintético",
                    color="red",
                    linestyle="-",
                )
                axs[0].set_yscale("log")
                axs[0].set_xlabel(variable)
                axs[0].set_ylabel("Frecuencia")
                axs[0].legend()
                axs[0].set_title(f"Distribución de {variable}")

                # Graficar error relativo en el segundo gráfico de fig
                axs[1].plot(
                    (edges[:-1] + edges[1:]) / 2,
                    100 * (hist_original - hist_synthetic) / hist_original,
                    color="black",
                )
                axs[1].set_ylabel("Error relativo (\%)")
                axs[1].axhline(0, color="gray", linestyle="--", linewidth=1)

                # Guardar la figura
                filename = f"comparacion_{variable}.png"
                if folder is not None:
                    filename = folder + filename
                fig.savefig(filename)
                plt.close(fig)

            # Generar y guardar las figuras de comparación
            plot_comparison(
                hist_lethargy_original,
                hist_lethargy_synthetic,
                edges_lethargy_original,
                "letargia",
                self.folder,
            )
            plot_comparison(
                hist_x_original,
                hist_x_synthetic,
                edges_x_original,
                "x",
                self.folder,
            )
            plot_comparison(
                hist_y_original,
                hist_y_synthetic,
                edges_y_original,
                "y",
                self.folder,
            )
            plot_comparison(
                hist_mu_original,
                hist_mu_synthetic,
                edges_mu_original,
                "mu",
                self.folder,
            )
            plot_comparison(
                hist_phi_original,
                hist_phi_synthetic,
                edges_phi_original,
                "phi",
                self.folder,
            )

    def get_tally_flux_2_df(self, tally_name: str) -> pd.DataFrame:
        """
        Devuelve un DataFrame con el flujo espacial normalizado para un tally de tipo 'flux_*'.

        Args:
            tally_name (str): Nombre del tally a procesar.

        Returns:
            pd.DataFrame: DataFrame con columnas ['z_midpoints', 'mean_norm', 'std.dev._norm'] y más información.
        """
        # Abrimos el archivo statepoint y accedemos al tally solicitado
        sp = openmc.StatePoint(self.folder + self.statepoint_name)
        tally = sp.get_tally(name=tally_name)
        df = tally.get_pandas_dataframe()

        # Renombrar columnas dependiendo si tiene desviación estándar
        if len(df.columns) == 7:
            df.columns = ["x", "y", "z", "nuclide", "score", "mean", "std.dev."]
        else:
            df.columns = ["x", "y", "z", "nuclide", "score", "mean"]
            df["std.dev."] = 1000.0  # Valor dummy para mantener estructura

        # Calcular volumen de celda
        volumen_celda = (
            tally.filters[0]._mesh.total_volume / tally.filters[0]._mesh.num_mesh_cells
        )

        # Normalizar mean y std.dev. por volumen de celda y factor de normalización
        df["mean_norm"] = df["mean"] / volumen_celda * self.factor_normalizacion
        df["std.dev._norm"] = df["std.dev."] / volumen_celda

        # Calcular puntos medios de z
        z_min = tally.filters[0]._mesh.lower_left[2]
        z_max = tally.filters[0]._mesh.upper_right[2]
        z = np.linspace(z_min, z_max, tally.filters[0]._mesh.num_mesh_cells + 1)
        z_midpoints = (z[:-1] + z[1:]) / 2
        df["z_midpoints"] = z_midpoints

        return df

    def get_tally_espectro_2_df(self, tally_name: str) -> pd.DataFrame:
        """
        Devuelve un DataFrame con el espectro energético normalizado para un tally de tipo 'espectro_*'.

        Args:
            tally_name (str): Nombre del tally a procesar.

        Returns:
            pd.DataFrame: DataFrame con columnas ['E_mid', 'mean_norm', 'std.dev._norm'] y más información.
        """
        sp = openmc.StatePoint(self.folder + self.statepoint_name)
        tally = sp.get_tally(name=tally_name)
        df = tally.get_pandas_dataframe()

        # Renombramos columnas para espectro (usa energía mínima y máxima)
        df.columns = [
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

        # Volumen de celda
        volumen_celda = (
            tally.filters[0]._mesh.total_volume / tally.filters[0]._mesh.num_mesh_cells
        )

        # Normalización
        df["mean_norm"] = df["mean"] / volumen_celda * self.factor_normalizacion
        df["std.dev._norm"] = df["std.dev."] / volumen_celda

        # Punto medio de energía
        df["E_mid"] = (df["E_min"] + df["E_max"]) / 2

        return df


def plot_flux_comparison(
    sim_original, sim_sintetica, save=True, filename="resultados_flujo.png"
) -> None:
    """
    Compara gráficamente el flujo entre simulaciones original y sintética para las regiones:
    total, agua y vacío, e incluye errores relativos (en %), interpolando sobre un eje z común.

    El dominio va desde z0_sint hasta el mínimo z_max de ambas curvas.
    """

    def compute_relative_error(
        df_orig, df_sint, z0_sint: float, num_interp_points: int = 1000
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calcula el error relativo entre los perfiles original y sintético de flujo,
        interpolando ambos sobre una grilla común en z.
        """
        z_min = z0_sint
        z_max = min(df_orig["z_midpoints"].max(), df_sint["z_midpoints"].max())
        z_interp = np.linspace(z_min, z_max, num_interp_points)

        interp_orig = np.interp(z_interp, df_orig["z_midpoints"], df_orig["mean_norm"])
        interp_sint = np.interp(z_interp, df_sint["z_midpoints"], df_sint["mean_norm"])

        # Crear máscara para valores no cero
        mask_nonzero = interp_orig != 0

        # Inicializar el array de error relativo con NaN
        err_rel = np.full_like(interp_orig, fill_value=np.nan)

        # Calcular el error relativo donde interp_orig no es cero
        err_rel[mask_nonzero] = (
            100
            * (interp_orig[mask_nonzero] - interp_sint[mask_nonzero])
            / interp_orig[mask_nonzero]
        )

        # Asignar infinito donde interp_orig es cero
        err_rel[~mask_nonzero] = np.inf

        return z_interp, err_rel
    
    def redondear_a_una_cifra(x: float) -> float:
        """
        Redondea un número positivo a una sola cifra significativa.
        Por ejemplo:
        - 3.76 → 4
        - 0.084 → 0.08
        - 123 → 100
        """
        if x == 0:
            return 0
        exponente = np.floor(np.log10(abs(x)))
        return round(x / 10**exponente) * 10**exponente

    # Obtener DataFrames de cada región
    df_total_orig = sim_original.get_tally_flux_2_df("flux_total")
    df_total_sint = sim_sintetica.get_tally_flux_2_df("flux_total")
    df_vacio_orig = sim_original.get_tally_flux_2_df("flux_vacio")
    df_vacio_sint = sim_sintetica.get_tally_flux_2_df("flux_vacio")
    df_agua_orig = get_tally_agua(df_total_orig, df_vacio_orig)
    df_agua_sint = get_tally_agua(df_total_sint, df_vacio_sint)

    fig = plt.figure(figsize=(16 * 1.25, 9 * 0.8))
    gs = gridspec.GridSpec(2, 3, height_ratios=[2.5, 1])
    axs = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]

    plt.subplots_adjust(
        left=0.07, right=0.95, top=0.94, bottom=0.00, wspace=0.3, hspace=0.25
    )

    nombres = ["Total", "Agua", "Vacío"]
    df_orig_all = [df_total_orig, df_agua_orig, df_vacio_orig]
    df_sint_all = [df_total_sint, df_agua_sint, df_vacio_sint]

    for i in range(3):
        df_orig = df_orig_all[i]
        df_sint = df_sint_all[i]

        # Establecer límites comunes en eje x
        z_min = 0
        z_max = max(df_orig["z_midpoints"].max(), df_sint["z_midpoints"].max())
        z_min = z_min - (z_max - z_min) * 0.05
        z_max = z_max + (z_max - z_min) * 0.05

        # Curvas originales y sintéticas
        axs[i].plot(
            df_orig["z_midpoints"],
            df_orig["mean_norm"],
            label="Original",
            color="blue",
            linestyle="--",
        )
        axs[i].plot(
            df_sint["z_midpoints"],
            df_sint["mean_norm"],
            label="Sintético",
            color="red",
            linestyle="-",
        )
        axs[i].set_title(f"Flujo {nombres[i]} vs. z")
        axs[i].set_ylabel(r"$\phi$ [cm$^{-2}$ s$^{-1}$]")
        axs[i].set_yscale("log")
        axs[i].set_xlabel("z [cm]")
        axs[i].grid(True, which="both", linestyle="--", linewidth=0.5)
        axs[i].legend()
        axs[i].set_xlim(z_min, z_max)  # sincronizar eje x

        # Error relativo interpolado
        z_interp, err_rel = compute_relative_error(df_orig, df_sint, sim_sintetica.z0)
        axs[i + 3].plot(z_interp, err_rel, color="black")
        axs[i + 3].axhline(0, color="gray", linestyle="--", linewidth=1)
        axs[i + 3].set_ylabel("Error relativo [\%]")

        # Cálculo de límites del eje y con margen simétrico (rango total)
        err_min = np.nanmin(err_rel)
        err_max = np.nanmax(err_rel)
        y_range = err_max - err_min
        y_margin = y_range * 0.05

        y_lim_min = max(err_min - y_margin, -100)
        y_lim_max = min(err_max + y_margin, 100)
        axs[i + 3].set_ylim(y_lim_min, y_lim_max)

        # Calcular paso = 20% del rango, redondeado a 1 cifra significativa
        paso_base = 0.2 * (y_lim_max - y_lim_min)

        # Redondear a 1 cifra significativa
        paso = max(redondear_a_una_cifra(paso_base), 1e-6)

        # Establecer ticks manuales
        ticks_y = np.arange(np.floor(y_lim_min), np.ceil(y_lim_max) + paso, paso)

        # Asegurarse de que 0 esté en los ticks
        if not np.any(np.isclose(ticks_y, 0)):
            ticks_y = np.sort(np.append(ticks_y, 0.0))

        axs[i + 3].set_yticks(ticks_y)
        
        axs[i + 3].grid(True, which="both", linestyle="--", linewidth=0.5)
        axs[i + 3].set_xlim(z_min, z_max)  # sincronizar eje x

    if save:
        full_path = os.path.join(sim_sintetica.folder, filename)
        plt.savefig(full_path, bbox_inches="tight", dpi=300)
    plt.show()


def plot_espectro_comparison(
    sim_original, sim_sintetica, z: float, save: bool = True, filename: str = None
) -> None:
    """
    Genera una figura con 6 subplots comparando el espectro (total, agua, vacío) original vs. sintético
    a una posición z dada, junto con sus errores relativos.

    Parámetros
    ----------
    sim_original : Simulation
        Simulación con los resultados originales.
    sim_sintetica : Simulation
        Simulación con los resultados sintéticos generados.
    z : float
        Posición z en cm correspondiente al tally de espectro.
    save : bool, opcional
        Si es True, guarda la figura como imagen PNG (default = True).
    filename : str, opcional
        Nombre del archivo PNG a guardar (default = "resultados_espectro_{z}.png").
    """

    def compute_relative_error(df_orig, df_sint) -> np.ndarray:
        """
        Calcula el error relativo porcentual entre dos series de espectro normalizado.
        """
        return (
            100 * (df_orig["mean_norm"] - df_sint["mean_norm"]) / df_orig["mean_norm"]
        )

    # Obtener DataFrames normalizados desde ambas simulaciones
    df_total_orig = sim_original.get_tally_espectro_2_df(f"espectro_total_{z}cm")
    df_total_sint = sim_sintetica.get_tally_espectro_2_df(f"espectro_total_{z}cm")
    df_vacio_orig = sim_original.get_tally_espectro_2_df(f"espectro_vacio_{z}cm")
    df_vacio_sint = sim_sintetica.get_tally_espectro_2_df(f"espectro_vacio_{z}cm")

    df_agua_orig = get_tally_agua(df_total_orig, df_vacio_orig)
    df_agua_sint = get_tally_agua(df_total_sint, df_vacio_sint)

    # Preparar figura y subplots: 2 filas (espectros, errores), 3 columnas (total, agua, vacío)
    fig, axs = plt.subplots(2, 3, figsize=(16 * 1.25, 9 * 0.8), sharex=f"col")
    plt.subplots_adjust(
        left=0.07, right=0.95, top=0.97, bottom=0.05, wspace=0.3, hspace=0.15
    )

    # Listas con nombres, dataframes, y etiquetas
    nombres = ["Total", "Agua", "Vacío"]
    df_orig_all = [df_total_orig, df_agua_orig, df_vacio_orig]
    df_sint_all = [df_total_sint, df_agua_sint, df_vacio_sint]

    for i in range(3):
        x = df_orig_all[i]["E_mid"]
        y_orig = df_orig_all[i]["mean_norm"]
        y_sint = df_sint_all[i]["mean_norm"]
        error_rel = compute_relative_error(df_orig_all[i], df_sint_all[i])

        # Fila superior: espectros
        axs[0, i].plot(x, y_orig, label="Original", color="blue", linestyle="--")
        axs[0, i].plot(x, y_sint, label="Sintético", color="red", linestyle="-")
        axs[0, i].set_xscale("log")
        axs[0, i].set_yscale("log")
        axs[0, i].set_title(f"Espectro {nombres[i]} en z = {z} cm")
        axs[0, i].set_ylabel("$\phi$ [cm$^{-2}$ s$^{-1}$ eV$^{-1}$]")
        axs[0, i].grid(True, which="both", linestyle="--")
        axs[0, i].legend()

        # Fila inferior: error relativo
        if np.any(~np.isinf(error_rel) & ~np.isnan(error_rel)):  # Verifica si hay algún número que no sea infinito ni NaN en error_rel
            axs[1, i].plot(x, error_rel, color="black")
            axs[1, i].axhline(0, color="gray", linestyle="--", linewidth=1)
            axs[1, i].set_xscale("log")
            axs[1, i].set_ylabel("Error relativo [%]")
            axs[1, i].set_ylim(max(error_rel.min(), -100), min(error_rel.max(), 100))
            axs[1, i].grid(True, which="both", linestyle="--")
        else:
            axs[1, i].set_visible(False)  # Si no hay valores útiles, deja el subplot vacío

    axs[1, 1].set_xlabel("Energía [eV]")

    # Nombre del archivo si no se especificó
    if filename is None:
        filename = f"resultados_espectro_{z:.0f}.png"

    if save:
        full_path = os.path.join(sim_sintetica.folder, filename)
        plt.savefig(full_path, bbox_inches="tight", dpi=300)

    plt.show()

def plot_espectro_comparison_all_common_z(sim_original, sim_sintetica, save=True) -> None:
    """
    Compara los espectros entre dos simulaciones en todos los valores de z que tienen en común.

    Parameters
    ----------
    sim_original : Simulation
        Simulación original.
    sim_sintetica : Simulation
        Simulación con fuente sintética.
    save : bool
        Si es True, guarda las imágenes generadas.
    """
    # Convertimos a conjuntos para buscar intersección
    z_orig = set(sim_original.z_for_spectral_tally)
    z_sint = set(sim_sintetica.z_for_spectral_tally)

    # z comunes entre ambas simulaciones
    z_comunes = sorted(z_orig.intersection(z_sint))

    if not z_comunes:
        print("No hay valores de z en común entre ambas simulaciones.")
        return

    print(f"Se encontraron {len(z_comunes)} valores de z en común: {z_comunes}")

    for z in z_comunes:
        filename = f"resultados_espectro_{z:.0f}cm.png"
        plot_espectro_comparison(
            sim_original=sim_original,
            sim_sintetica=sim_sintetica,
            z=z,
            save=save,
            filename=filename,
        )



def get_tally_agua(df_total, df_vacio):
    volumen_total = df_total["mean"].sum() / df_total["mean_norm"].sum()
    volumen_vacio = df_vacio["mean"].sum() / df_vacio["mean_norm"].sum()
    volumen_agua = volumen_total - volumen_vacio

    df_agua = df_total.copy()
    df_agua["mean"] = df_agua["mean"] - df_vacio["mean"]
    df_agua["std.dev."] = np.sqrt(
        df_agua["std.dev."] ** 2 + df_vacio["std.dev."] ** 2
    )  # Revisar esto si hace falta. No se reviso hasta el momento.
    df_agua["mean_norm"] = df_agua["mean"] / volumen_agua
    df_agua["std.dev._norm"] = df_agua["std.dev."] / volumen_agua

    return df_agua

