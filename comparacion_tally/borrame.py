import sys

sys.path.append("/home/lucas/Documents/Proyecto_Integrador/PI")
from functions import *

factor_normalizacion = 2.615/2.624 *2.614 / 2.61 *8.424/4631/0.08257*3.336
z_min = 140

# Load the statepoint file 1
sp = openmc.StatePoint("statepoint_original.h5")
tally = sp.get_tally(name="flux")
df = tally.get_pandas_dataframe()
df.columns = ["x", "y", "z", "nuclide", "score", "mean", "std.dev."]
plt.plot(df["z"][z_min:] * 100 / 750, df["mean"][z_min:], label="Original")

df1 = df

# Load the statepoint file 2
sp = openmc.StatePoint("statepoint_sintetico.h5")
tally = sp.get_tally(name="flux")
df = tally.get_pandas_dataframe()
df.columns = ["x", "y", "z", "nuclide", "score", "mean", "std.dev."]
plt.plot(
    df["z"][z_min:] * 100 / 750,
    df["mean"][z_min:] * factor_normalizacion * 3.031 / 2.278 * 0.264 / 3.306,
    label="Sintetico",
)
plt.xlabel("z [cm]")
plt.yscale("log")
plt.grid()
plt.title("Flux vs. z")
plt.legend()

plt.show()

# error relativo
plt.figure()
plt.plot(
    df1["z"][z_min:] * 100 / 750,
    (
        df1["mean"][z_min:]
        - df["mean"][z_min:] * factor_normalizacion * 3.031 / 2.278 * 0.264 / 3.306
    )
    / df1["mean"][z_min:],
)
plt.xlabel("z [cm]")
plt.grid()
plt.title("Error relativo")
plt.show()
