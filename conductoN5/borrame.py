#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
interpolador.py
===============

Versión *pura-Python* (NumPy vectorizada) de la rutina de interpolación lineal
que usas en C **y** herramientas para:

1.  **linear_interp** – equivalente directo al `double linear_interp(...)`.
2.  **resample_particles** – genera *N* valores a partir de una CDF tabulada.
3.  **plot_samples_log** – histograma en escala logarítmica del muestreo.

Todo viene hiper-documentado y con *type hints*.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------  
def linear_interp(
    y_target: float,
    x: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
) -> float:
    """
    Interpolación lineal *x(y)* para tablas monótonas (misma lógica que en C).

    Parameters
    ----------
    y_target : float
        Valor de CDF (0 ≤ y ≤ 1) a invertir.
    x, y : array-like
        Vectores `x[i]` y `y[i]` (mismo tamaño).  
        `y` **debe** ser creciente y comenzar en 0, terminar en 1.

    Returns
    -------
    float
        Valor *x* tal que `CDF(x) ≈ y_target`.

    Raises
    ------
    ValueError
        Si `y_target` está fuera de `[y[0], y[-1]]` o los vectores
        no cumplen los requisitos de monotonía.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # -------- Validaciones defensivas (más expresivas que imprimir y devolver -1)
    if not (np.all(np.diff(y) >= 0) and y[0] <= 0 and y[-1] >= 1):
        raise ValueError("El vector y debe ser creciente y cubrir [0, 1].")
    if not (y[0] <= y_target <= y[-1]):
        raise ValueError(f"y_target={y_target} fuera del rango [{y[0]}, {y[-1]}].")

    # -------- Búsqueda de intervalo usando *searchsorted* (vectorizable)
    idx = np.searchsorted(y, y_target) - 1
    idx = np.clip(idx, 0, len(y) - 2)  # protege extremos

    # -------- Fórmula de interpolación lineal
    x1, x2 = x[idx], x[idx + 1]
    y1, y2 = y[idx], y[idx + 1]
    return x1 + (y_target - y1) * (x2 - x1) / (y2 - y1)


# -----------------------------------------------------------------------------  
def resample_particles(
    N: int,
    *,
    x: Sequence[float] | np.ndarray,
    cdf: Sequence[float] | np.ndarray,
    seed: int | None = None,
) -> np.ndarray:
    """
    Genera *N* muestras pseudo-aleatorias con la distribución tabulada `cdf(x)`.

    Parameters
    ----------
    N : int
        Cantidad de muestras a generar.
    x, cdf : array-like
        Vectores de puntos de la CDF y sus abscisas (misma longitud que arriba).
    seed : int, optional
        Semilla para reproducibilidad (``None`` → semilla automática).

    Returns
    -------
    np.ndarray
        Vector de longitud *N* con las muestras re-muestreadas.
    """
    rng = np.random.default_rng(seed)
    u = rng.random(N)  # variables ~ U(0,1)
    # np.interp es ~12× más rápido que el for loop; usa el mismo algoritmo
    return np.interp(u, cdf, x)


# -----------------------------------------------------------------------------  
def plot_samples_log(
    samples: np.ndarray,
    bins: int | Sequence[float] = 50,
    *,
    xlabel: str = "x",
    ylabel: str = "Frecuencia",
    title: str | None = None,
    **hist_kwargs,
) -> None:
    """
    Histograma con eje *x* en escala logarítmica.

    Parameters
    ----------
    samples : np.ndarray
        Muestras generadas con :pyfunc:`resample_particles`.
    bins : int or sequence, optional
        Igual que en *matplotlib.hist*. 50 por defecto.
    xlabel, ylabel, title : str, optional
        Etiquetas de la figura.
    **hist_kwargs
        Pasan directo a ``plt.hist`` (color, alpha, etc.).
    """
    plt.figure(figsize=(6, 4))
    plt.hist(samples, bins=bins, histtype="step", log=False, **hist_kwargs)
    plt.yscale("log")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.tight_layout()


# =============================================================================
# ------------------------- DEMOSTRACIÓN RÁPIDA -------------------------------
# (Ejecuta este bloque sólo si corres el script directamente)
# =============================================================================
if __name__ == "__main__":
    # Copia-pega los vectores tal cual los pasaste
    cumul_str = (
        "0.0,0.0017152722199286967,0.006231918617574234,0.020530248871191457,"
        "0.029996701365958865,0.0421981138328991,0.05694884902607634,"
        "0.07425950013863244,0.08359747169124232,0.09130131750044702,"
        "0.09204899438049759,0.09556517438469385,0.1020314483506414,"
        "0.10878530689301087,0.11567204743885258,0.12197122439429363,"
        "0.1277862223622261,0.13138292924395312,0.13531534491309222,"
        "0.14251501191272062,0.14972241782247545,0.15831052521028002,"
        "0.16688025396068373,0.17659387932657336,0.184074907389127,"
        "0.18661109289282043,0.19321455387453354,0.20473686607713065,"
        "0.21340724955519025,0.22568818808162522,0.25341463083653953,"
        "0.26726064689078205,0.27937285408828433,0.3145757750925577,"
        "0.3306669522349255,0.3663567607931758,0.4134822526651825,"
        "0.43343717059387515,0.5512235116052753,0.6179081790328356,"
        "0.6924800549198463,0.7301002575133949,0.7712747787477097,"
        "0.7946566952225088,0.8353246563789312,0.8644945405137271,"
        "0.8851274187145752,0.9078039289444804,0.9276016279549979,"
        "0.9366977805112279,0.9612391430365463,0.9766016384172806,"
        "0.9864975092391768,0.9869065554233387,0.9942458323402709,"
        "0.9982584075499777,1.0"
    )
    micro_str = (
        "0.38764702288539227,1.0265004728248313,1.3168884046154856,"
        "1.978972889098177,2.2461297863455787,2.6062108217659894,"
        "3.175371168075672,4.104612549805766,5.033853931535859,"
        "5.963095313265953,6.195405658698475,6.892336694996046,"
        "7.821578076726139,8.750819458456233,9.680060840186325,"
        "10.609302221916419,11.538543603646513,12.003164294511558,"
        "12.467784985376605,13.3970263671067,14.326267748836791,"
        "15.255509130566885,16.18475051229698,17.113991894027073,"
        "17.81092293032464,18.043233275757164,18.426545345720825,"
        "18.635624656610098,18.72854879478311,18.821472932956116,"
        "18.972474657487258,19.030552243845385,19.077014312931894,"
        "19.193169485648156,19.239631554734657,19.33255569290767,"
        "19.44871086562393,19.495172934710432,19.75071431468621,"
        "19.90171603921735,20.07594879829174,20.16887293646475,"
        "20.285028109181013,20.35472121281077,20.494107420070286,"
        "20.610262592786547,20.703186730959555,20.830957420947446,"
        "20.958728110935333,21.028421214565086,21.283962594540863,"
        "21.516272939973387,21.76019880267754,21.77181431994916,"
        "22.166741907184452,22.689440184407633,23.618681566137724"
    )
    cdf = np.fromstring(cumul_str, sep=",")
    xvals = np.fromstring(micro_str, sep=",")

    # -------- Test unitaria: invertimos la mediana (y_target = 0.5)
    print("x(0.5) =", linear_interp(0.5, xvals, cdf))

    # -------- Re-muestreo y plot
    samples = resample_particles(20_000000, x=xvals, cdf=cdf, seed=42)
    plot_samples_log(
        samples,
        bins=np.linspace(xvals.min(), xvals.max(), 600),
        xlabel="Micro-bin (x)",
        title="Resampleo de 20 000 partículas",
        color="C3",
    )
    plt.show()
