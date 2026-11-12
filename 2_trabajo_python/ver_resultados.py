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
    04_estados.png / .pdf   (desplazamiento x^1_t y velocidad x^2_t estimados)
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

        # Trayectorias de estados x^1_t = s(t) y x^2_t = sdot(t), si existen.
        # Las corridas anteriores a esta version no las guardaban.
        estados = None
        if {"x1", "x2", "iteracion_estado"} <= set(ds.variables):
            estados = {
                "iteracion": np.asarray(
                    ds.variables["iteracion_estado"][:], dtype=int
                ),
                "x1": np.asarray(ds.variables["x1"][:], dtype=float),
                "x2": np.asarray(ds.variables["x2"][:], dtype=float),
            }

    print(f"  Posterior leido de: {ruta}")
    return muestras, estados


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


def figura_estados(tiempo, y, estados, quema):
    """Estados estimados, en el espiritu de la Figura 5 del articulo.

    Usa las trayectorias x_{0:T} guardadas por PMH despues del burn-in, que
    aproximan la distribucion de suavizado p(x_{0:T} | y_{1:T}).

    Panel superior: desplazamiento x^1_t = s(t) con los datos simulados y_t.
    Panel inferior: velocidad x^2_t = sdot(t), que no se observa directamente.
    En ambos se muestra la media posterior y el intervalo de 95 %.
    """
    seleccion = estados["iteracion"] >= quema
    if not np.any(seleccion):
        raise ValueError("No hay trayectorias de estados despues de la quema.")

    x1 = estados["x1"][seleccion]
    x2 = estados["x2"][seleccion]
    tiempo_segundos = 0.1 * np.arange(x1.shape[1])

    fig, (ax_s, ax_v) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    # Desplazamiento x^1_t = s(t) frente a los datos simulados.
    ax_s.scatter(
        0.1 * tiempo, y, s=4, color="black", alpha=0.6,
        label=r"Datos simulados $y_t$", zorder=1,
    )
    ax_s.fill_between(
        tiempo_segundos,
        np.percentile(x1, 2.5, axis=0),
        np.percentile(x1, 97.5, axis=0),
        color="#4C9ED9", alpha=0.35,
        label=r"Intervalo 95 % de $p(x^1_t \mid y_{1:T})$", zorder=2,
    )
    ax_s.plot(
        tiempo_segundos, x1.mean(axis=0), color="#1F5F99", lw=1.0,
        label=r"Media posterior de $x^1_t = s(t)$", zorder=3,
    )
    ax_s.set_ylabel(r"desplazamiento $x^1_t = s(t)$")
    ax_s.grid(alpha=0.3)
    ax_s.legend(loc="upper right", fontsize=8)

    # Velocidad x^2_t = sdot(t): estado no observado.
    ax_v.fill_between(
        tiempo_segundos,
        np.percentile(x2, 2.5, axis=0),
        np.percentile(x2, 97.5, axis=0),
        color="#F28E2B", alpha=0.35,
        label=r"Intervalo 95 % de $p(x^2_t \mid y_{1:T})$",
    )
    ax_v.plot(
        tiempo_segundos, x2.mean(axis=0), color="#B5520B", lw=1.0,
        label=r"Media posterior de $x^2_t = \dot{s}(t)$",
    )
    ax_v.set_ylabel(r"velocidad $x^2_t = \dot{s}(t)$")
    ax_v.set_xlabel("tiempo (s)")
    ax_v.grid(alpha=0.3)
    ax_v.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    rutas = guardar_figura(fig, "04_estados")
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
    muestras, estados = cargar_posterior()

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
    if estados is None:
        print(
            "  Aviso: el posterior no contiene trayectorias de estados; "
            "no se genera 04_estados.\n"
            "  Vuelva a ejecutar pmh_damper.py con la version actual."
        )
    else:
        n_tray = int(np.sum(estados["iteracion"] >= quema))
        print(f"  Trayectorias de estados despues de la quema: {n_tray}")
        archivos.extend(figura_estados(tiempo, y, estados, quema))
    df, ruta_csv = resumen(muestras, verdaderos, quema)
    archivos.append(ruta_csv)

    print("\nResumen posterior:")
    print(df.to_string(index=False))

    print("\nArchivos generados:")
    for ruta in archivos:
        print("  " + ruta)


if __name__ == "__main__":
    main()
