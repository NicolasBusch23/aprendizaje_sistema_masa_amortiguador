# -*- coding: utf-8 -*-
"""
Genera las figuras y el resumen del ejemplo "Damper" (amortiguador no lineal).

Usa:
    data/obs.nc -> datos simulados publicados con el articulo
    resultados_python/posterior_python_corregido.nc -> posterior PMH corregido

NOTA: ``results/posterior.nc`` del paquete original contiene solo valores de
relleno. La reproduccion Python se guarda aparte para conservar el original.

Salidas (carpeta figuras/):
    01_datos_observados.png / .pdf
    02_posteriores.png / .pdf
    03_trazas.png / .pdf
    resumen_posterior.csv
"""

import os
from math import lgamma

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
# Fuentes TrueType incrustadas: conservan calidad y facilitan editar el PDF.
matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt
from netCDF4 import Dataset

AQUI = os.path.dirname(os.path.abspath(__file__))
os.chdir(AQUI)
FIG = "figuras"
os.makedirs(FIG, exist_ok=True)

PARAMS = ["f_c", "c_0", "k", "p"]
ETIQUETAS = {
    "f_c": r"$f_c$ (friccion de Coulomb)",
    "c_0": r"$c_0$ (amortiguamiento viscoso)",
    "k": r"$k$ (rigidez del resorte)",
    "p": r"$p$ (exponente no lineal)",
}

# Prioris de la seccion 7 y del apendice Damper.bi.
# LibBi y NumPy usan la parametrizacion Gamma(forma, escala).
PRIOR_GAMMA = {
    "f_c": (2.0, 0.01),
    "c_0": (2.0, 1.0),
    "k": (4.0, 0.3),
}

# Limites usados en la Figura 4 del articulo. En el panel de p el articulo
# prolonga el eje hasta 2, aunque la prior U(0,1) vale cero fuera de [0,1].
LIMITES_X = {
    "f_c": (0.0, 0.1),
    "c_0": (0.0, 3.0),
    "k": (0.0, 4.0),
    "p": (0.0, 2.0),
}

QUEMA_FRAC = 0.2  # fraccion inicial descartada como burn-in


def _leer_variable(ds, nombre):
    """Lee una variable NetCDF respetando mascara y _FillValue de LibBi."""
    arr = ds.variables[nombre][:]
    datos = np.ma.getdata(arr).astype(float)
    mascara = np.ma.getmaskarray(arr)
    relleno = getattr(ds.variables[nombre], "_FillValue", None)
    if relleno is not None:
        mascara = mascara | np.isclose(datos, float(relleno))
    datos[mascara] = np.nan
    return datos


def cargar_observaciones():
    """Recolecta las observaciones desde obs.nc (misma logica que pmh_damper)."""
    ds = Dataset(os.path.join("data", "obs.nc"), "r")
    try:
        tiempo = _leer_variable(ds, "time").ravel()
        y_raw = _leer_variable(ds, "y")
        if y_raw.ndim == 1:
            y = y_raw
        else:
            dims = ds.variables["y"].dimensions
            eje_np = dims.index("np") if "np" in dims else y_raw.ndim - 1
            y = np.take(y_raw, 0, axis=eje_np).ravel()
        verdaderos = {}
        for p in PARAMS:
            if p in ds.variables:
                verdaderos[p] = float(
                    np.ma.getdata(ds.variables[p][:]).ravel()[0])
    finally:
        ds.close()
    return tiempo, y, verdaderos


def cargar_posterior():
    """Carga primero la corrida corregida; conserva compatibilidad con la vieja."""
    candidatos = [
        os.path.join("resultados_python", "posterior_python_corregido.nc"),
        os.path.join("resultados_python", "posterior_python.nc"),
    ]
    ruta = next((r for r in candidatos if os.path.exists(r)), None)
    if ruta is None:
        raise SystemExit(
            "No existe una cadena posterior.\n"
            "Ejecuta: python pmh_damper.py --iter 10000 --part 256"
        )

    with Dataset(ruta, "r") as ds:
        muestras = {
            p: np.asarray(ds.variables[p][:], dtype=float) for p in PARAMS
        }

    print(f"  Posterior leido de: {ruta}")
    return muestras


def guardar_figura(fig, nombre):
    """Guarda una figura como PNG de 300 dpi y como PDF vectorial."""
    ruta_png = os.path.join(FIG, f"{nombre}.png")
    ruta_pdf = os.path.join(FIG, f"{nombre}.pdf")
    fig.savefig(ruta_png, dpi=300, bbox_inches="tight")
    fig.savefig(ruta_pdf, format="pdf", bbox_inches="tight")
    return [ruta_png, ruta_pdf]


def figura_datos(tiempo, y):
    # El eje de obs.nc es el indice 0,...,1000. Con T_s=0.1, el tiempo fisico
    # de la Figura 3 del articulo es 0,...,100 segundos.
    tiempo_segundos = 0.1 * tiempo
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(tiempo_segundos, y, lw=0.7, color="#1f77b4")
    ax.set_xlabel("tiempo (s)")
    ax.set_ylabel("posicion observada  y")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    rutas = guardar_figura(fig, "01_datos_observados")
    plt.close(fig)
    return rutas


def densidad_priori(parametro, x):
    """Evalua las densidades prior definidas en el articulo.

    Para f_c, c_0 y k se usa Gamma(forma, escala). Para p se usa
    Uniforme(0,1). La funcion devuelve cero fuera del soporte.
    """
    x = np.asarray(x, dtype=float)

    if parametro == "p":
        return np.where((x >= 0.0) & (x <= 1.0), 1.0, 0.0)

    forma, escala = PRIOR_GAMMA[parametro]
    densidad = np.zeros_like(x)
    soporte = x > 0.0
    xs = x[soporte]
    log_densidad = (
        (forma - 1.0) * np.log(xs)
        - xs / escala
        - forma * np.log(escala)
        - lgamma(forma)
    )
    densidad[soporte] = np.exp(log_densidad)
    return densidad


def figura_posteriores(muestras, verdaderos, quema):
    """Reproduce la estructura de la Figura 4 del articulo.

    Cada panel contiene la marginal posterior (histograma azul), la densidad
    prior (curva negra) y el valor verdadero usado para simular (linea roja).
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    for ax, parametro in zip(axes.ravel(), PARAMS):
        posterior = np.asarray(muestras[parametro][quema:], dtype=float)
        posterior = posterior[np.isfinite(posterior)]
        if posterior.size == 0:
            raise ValueError(
                f"No hay muestras posteriores validas para {parametro} "
                f"despues de la quema ({quema})."
            )

        xmin, xmax = LIMITES_X[parametro]
        rejilla = np.linspace(xmin, xmax, 1000)

        ax.hist(
            posterior,
            bins=40,
            range=(xmin, xmax),
            density=True,
            color="#4C9ED9",
            alpha=0.85,
            edgecolor="white",
            linewidth=0.4,
            label="Posterior",
        )
        ax.plot(
            rejilla,
            densidad_priori(parametro, rejilla),
            color="black",
            linewidth=2.0,
            label="Prior",
        )

        valor_verdadero = verdaderos.get(parametro)
        if valor_verdadero is not None and np.isfinite(valor_verdadero):
            ax.axvline(
                valor_verdadero,
                color="#D62728",
                linewidth=2.0,
                label=f"Valor verdadero = {valor_verdadero:.4g}",
            )

        ax.set_xlim(xmin, xmax)
        ax.set_title(ETIQUETAS[parametro])
        ax.set_xlabel("valor del parametro")
        ax.set_ylabel("densidad de probabilidad")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    fig.tight_layout()
    rutas = guardar_figura(fig, "02_posteriores")
    plt.close(fig)
    return rutas


def figura_trazas(muestras, verdaderos):
    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    for ax, p in zip(axes, PARAMS):
        ax.plot(muestras[p], lw=0.5, color="#555555", label="Cadena MCMC")
        vv = verdaderos.get(p)
        if vv is not None:
            ax.axhline(vv, color="#d62728", ls="--", lw=1.4,
                       label="Valor verdadero")
        ax.set_ylabel(p)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("iteracion MCMC")
    fig.tight_layout()
    rutas = guardar_figura(fig, "03_trazas")
    plt.close(fig)
    return rutas


def resumen(muestras, verdaderos, quema):
    filas = []
    for p in PARAMS:
        x = muestras[p][quema:]
        filas.append({
            "parametro": p,
            "verdadero": verdaderos.get(p),
            "media": x.mean(),
            "mediana": np.median(x),
            "desv_est": x.std(ddof=1),
            "IC_2.5%": np.percentile(x, 2.5),
            "IC_97.5%": np.percentile(x, 97.5),
        })
    df = pd.DataFrame(filas)
    ruta = os.path.join(FIG, "resumen_posterior.csv")
    df.to_csv(ruta, index=False)
    return df, ruta


def main():
    print("Leyendo archivos...")
    tiempo, y, verdaderos = cargar_observaciones()
    muestras = cargar_posterior()

    n = len(muestras["f_c"])
    quema = int(n * QUEMA_FRAC)
    cadena_post = np.column_stack([muestras[p][quema:] for p in PARAMS])
    n_unicos = np.unique(cadena_post, axis=0).shape[0]

    print(f"  Observaciones: {y.size}")
    print(f"  Muestras posteriores: {n} (quema = {quema})")
    print(f"  Vectores theta distintos despues de la quema: {n_unicos}")
    if n < 10_000 or n_unicos < 20:
        print(
            "  ADVERTENCIA: esta cadena no reproduce todavia la corrida del "
            "articulo (10000 muestras).\n"
            "  El grafico se genera para inspeccion, pero no debe interpretarse "
            "como resultado final."
        )

    print("\nGenerando figuras...")
    archivos = []
    archivos.extend(figura_datos(tiempo, y))
    archivos.extend(figura_posteriores(muestras, verdaderos, quema))
    archivos.extend(figura_trazas(muestras, verdaderos))
    df, ruta_csv = resumen(muestras, verdaderos, quema)
    archivos.append(ruta_csv)

    print("\nResumen posterior:")
    print(df.to_string(index=False))

    print("\nArchivos generados:")
    for ruta in archivos:
        print("  " + ruta)


if __name__ == "__main__":
    main()
