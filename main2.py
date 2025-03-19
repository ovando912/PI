import sys

sys.path.append("/home/lucas/Documents/Proyecto_Integrador/PI")
from functions import *

folder_path = "/home/lucas/Documents/Proyecto_Integrador/PI/segundo_semestre/3-17-25"
os.chdir(folder_path)

geometria = [True, 15, 15, 100, 3, 3]
z0 = 30
columns_order = ["ln(E0/E)", "x", "y", "mu", "phi"]

# Definir los argumentos necesarios
folder = "/home/lucas/Documents/Proyecto_Integrador/PI/segundo_semestre/3-17-25/"
sample_count = "50000"  # Nota: se pasa como cadena, ya que se envía desde la línea de comandos
source = "source.xml"
result = "sint_source.mcpl"
statepoint_name = "statepoint_sintetico.h5"

batches = 2

for i in range(batches):
    print(f"Corriendo simulación {i + 1} de {batches}")

    # Construir la lista de argumentos para el ejecutable
    # Según nuestro ejemplo, se usan las opciones -f (folder), -n (cantidad de sampleos),
    # -s (nombre del archivo source) y -r (nombre del archivo de salida)
    cmd = ["/home/lucas/Documents/Proyecto_Integrador/PI/main", "-f", folder, "-n", sample_count, "-s", source, "-r", generate_filename(result, i+2)]

    try:
        # Ejecuta el programa y captura la salida
        result_process = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("Salida del programa:")
        print(result_process.stdout)
    except subprocess.CalledProcessError as e:
        print("El programa falló con el siguiente error:")
        print(e.stderr)

    print(generate_filename(result, i))
    SurfaceSourceFile = kds.SurfaceSourceFile(generate_filename(result, i+2)+".gz")
    df = SurfaceSourceFile.get_pandas_dataframe()
    cantidad = len(df)
    # plot_correlated_variables(df, columns_order, filename="sintetico.png")
    del SurfaceSourceFile
    kds.create_source_file(df, generate_filename("sint_source.h5",i+2))
    del df
    fuente_sintetica = [generate_filename("sint_source.h5",i+2)]
    run_simulation(fuente_sintetica, geometria, z0, cantidad, generate_filename(statepoint_name, i+2))