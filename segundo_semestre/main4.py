import sys

sys.path.append("/home/lucas/Documents/Proyecto_Integrador/PI")
from functions import *

path = "/home/lucas/Documents/Proyecto_Integrador/PI/segundo_semestre/sources/source2"

os.chdir(path)

# Inputs:

geometria = [True, 15, 15, 100, 3, 3]
z0 = 30
fuente_original = ["monoenergetica", "colimada"]
N_original = int(1e6)

run_simulation(fuente_original, geometria, z0, N_original)