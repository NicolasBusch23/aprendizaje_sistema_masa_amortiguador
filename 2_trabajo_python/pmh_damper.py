# -*- coding: utf-8 -*-
"""Inferencia PMH para el ejemplo del amortiguador no lineal.

Este archivo implementa:

* el modelo de estado de las ecuaciones (41)-(42) del articulo;
* el filtro de particulas bootstrap del Algoritmo 2;
* particle Metropolis-Hastings (PMH) del Algoritmo 3;
* la propuesta gaussiana truncada declarada en ``Damper.bi``;
* la extraccion de trayectorias de estados x_{0:T} dentro de PMH, que
  aproxima la distribucion de suavizado p(x_{0:T} | y_{1:T}) de la Figura 5.

Referencia:
    Schon, Svensson, Murray y Lindsten (2018), "Probabilistic learning of
    nonlinear dynamical systems using sequential Monte Carlo".

Decisiones de reproduccion
--------------------------
1. Se usa ``m=2`` porque ese es el valor del apendice ``Damper.bi`` y el que
   reproduce ``obs.nc``. El cuerpo del articulo dice ``m=8``; es una
   contradiccion del documento.
2. LibBi recibe desviaciones estandar en ``gaussian``. Por tanto,
   ``sigma_v=0.01`` y ``sigma_y=0.1``, como confirma la seccion 7 al escribir
   N(0, 0.01^2) y N(0, 0.1^2).
3. El signo reproduce el ternario de LibBi: signo(0)=+1.
4. ``obs.nc`` incluye la observacion del estado inicial en el indice 0 y luego
   una observacion por transicion, hasta el indice 1000.
"""

from __future__ import annotations

import argparse
import os
from math import lgamma
from time import perf_counter

import numpy as np
from netCDF4 import Dataset
from scipy.special import ndtr, ndtri

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
AQUI = os.path.dirname(os.path.abspath(__file__))
os.chdir(AQUI)
CARPETA_SALIDA = "resultados_python"
os.makedirs(CARPETA_SALIDA, exist_ok=True)

# ---------------------------------------------------------------------------
# Modelo: ecuaciones (41)-(42) y apendice Damper.bi
# ---------------------------------------------------------------------------
T_S = 0.1
MASA = 2.0
SIGMA_PROCESO = 1.0e-2       # v_t ~ N(0, 0.01^2)
SIGMA_OBSERVACION = 1.0e-1   # e_t ~ N(0, 0.1^2)
S_INICIAL = 0.5
VELOCIDAD_INICIAL = 0.0

# Se conserva el orden del archivo LibBi.
PARAMETROS = ("f_c", "c_0", "k", "p")

# Gamma(forma, escala), tal como se aprecia en las prioris de la Figura 4.
PRIOR_GAMMA = {
    "f_c": (2.0, 0.01),
    "c_0": (2.0, 1.0),
    "k": (4.0, 0.3),
}

# Propuestas del bloque proposal_parameter de Damper.bi.
DESV_PROPUESTA = np.array([1.0e-3, 1.0e-2, 1.0e-2, 1.0e-2])
LIMITE_INFERIOR = np.array([0.0, 0.0, 0.0, 0.0])
LIMITE_SUPERIOR = np.array([np.inf, 1.0, np.inf, 1.0])

LOG_RAIZ_2PI = 0.5 * np.log(2.0 * np.pi)


# ---------------------------------------------------------------------------
# Datos LibBi
# ---------------------------------------------------------------------------
def _leer_variable(ds: Dataset, nombre: str) -> np.ndarray:
    """Convierte una variable NetCDF en un arreglo y conserva faltantes como NaN."""
    variable = ds.variables[nombre]
    arreglo = variable[:]
    datos = np.ma.getdata(arreglo).astype(float)
    mascara = np.ma.getmaskarray(arreglo)

    relleno = getattr(variable, "_FillValue", None)
    if relleno is not None:
        mascara = mascara | np.isclose(datos, float(relleno))

    datos[mascara] = np.nan
    return datos


def cargar_observaciones(ruta: str | None = None):
    """Carga tiempo, desplazamiento observado y parametros verdaderos."""
    ruta = ruta or os.path.join("data", "obs.nc")
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No se encuentra el archivo: {ruta}")

    with Dataset(ruta, "r") as ds:
        tiempo = _leer_variable(ds, "time").ravel()
        y_bruto = _leer_variable(ds, "y")

        if y_bruto.ndim == 1:
            y = y_bruto
        else:
            dimensiones = ds.variables["y"].dimensions
            eje_repeticion = (
                dimensiones.index("np") if "np" in dimensiones
                else y_bruto.ndim - 1
            )
            y = np.take(y_bruto, 0, axis=eje_repeticion).ravel()

        verdaderos = {
            nombre: float(np.ma.getdata(ds.variables[nombre][:]).ravel()[0])
            for nombre in PARAMETROS
            if nombre in ds.variables
        }

    if tiempo.size != y.size:
        raise ValueError(
            f"Longitudes incompatibles: time={tiempo.size}, y={y.size}."
        )
    if np.isnan(tiempo).any():
        raise ValueError("El eje temporal contiene valores faltantes.")

    return tiempo, y, verdaderos


# ---------------------------------------------------------------------------
# Prior p(theta), seccion 7
# ---------------------------------------------------------------------------
def log_gamma(x: float, forma: float, escala: float) -> float:
    """Log-densidad Gamma con parametrizacion forma-escala."""
    if x <= 0.0:
        return -np.inf
    return (
        (forma - 1.0) * np.log(x)
        - x / escala
        - forma * np.log(escala)
        - lgamma(forma)
    )


def log_priori(theta: np.ndarray) -> float:
    """Evalua p(f_c)p(c_0)p(k)p(p)."""
    f_c, c_0, k, p = theta
    if f_c <= 0.0 or c_0 <= 0.0 or k <= 0.0 or not (0.0 < p < 1.0):
        return -np.inf

    return (
        log_gamma(f_c, *PRIOR_GAMMA["f_c"])
        + log_gamma(c_0, *PRIOR_GAMMA["c_0"])
        + log_gamma(k, *PRIOR_GAMMA["k"])
        # U(0,1) tiene log-densidad 0 dentro de su soporte.
    )


def muestrear_priori(rng: np.random.Generator) -> np.ndarray:
    """Obtiene un inicio dentro de la region accesible por Damper.bi."""
    while True:
        theta = np.array([
            rng.gamma(*PRIOR_GAMMA["f_c"]),
            rng.gamma(*PRIOR_GAMMA["c_0"]),
            rng.gamma(*PRIOR_GAMMA["k"]),
            rng.uniform(0.0, 1.0),
        ])
        # Damper.bi limita las propuestas de c_0 y p al intervalo [0,1].
        if theta[1] <= 1.0:
            return theta


# ---------------------------------------------------------------------------
# Propuesta q(theta' | theta): gaussianas truncadas de Damper.bi
# ---------------------------------------------------------------------------
def _normalizacion_truncada(centro: np.ndarray) -> np.ndarray:
    """Constantes Z del normal truncado para cada componente de theta."""
    a = (LIMITE_INFERIOR - centro) / DESV_PROPUESTA
    b = (LIMITE_SUPERIOR - centro) / DESV_PROPUESTA
    z = ndtr(b) - ndtr(a)
    if np.any(z <= 0.0):
        raise FloatingPointError("Normalizacion nula en la propuesta truncada.")
    return z


def proponer_theta(theta: np.ndarray, rng: np.random.Generator):
    """Muestrea theta' y calcula log[q(theta|theta')/q(theta'|theta)].

    El cociente aparece en la ecuacion (33). Aunque el nucleo gaussiano es
    simetrico, las constantes de normalizacion cambian cerca de los limites.
    """
    z_actual = _normalizacion_truncada(theta)
    cdf_inferior = ndtr((LIMITE_INFERIOR - theta) / DESV_PROPUESTA)
    cdf_superior = ndtr((LIMITE_SUPERIOR - theta) / DESV_PROPUESTA)

    u = cdf_inferior + rng.random(theta.size) * (
        cdf_superior - cdf_inferior
    )
    epsilon = np.finfo(float).eps
    u = np.clip(u, epsilon, 1.0 - epsilon)
    propuesta = theta + DESV_PROPUESTA * ndtri(u)

    # Proteccion exclusiva contra redondeo en los extremos.
    propuesta = np.maximum(propuesta, np.nextafter(LIMITE_INFERIOR, np.inf))
    finitos = np.isfinite(LIMITE_SUPERIOR)
    propuesta[finitos] = np.minimum(
        propuesta[finitos],
        np.nextafter(LIMITE_SUPERIOR[finitos], -np.inf),
    )

    z_propuesta = _normalizacion_truncada(propuesta)
    log_cociente_q = float(np.sum(np.log(z_actual) - np.log(z_propuesta)))
    return propuesta, log_cociente_q


# ---------------------------------------------------------------------------
# Algoritmo 2: filtro de particulas bootstrap
# ---------------------------------------------------------------------------
def signo_libbi(x: np.ndarray) -> np.ndarray:
    """Convencion exacta del ternario de Damper.bi: +1 cuando x=0."""
    return np.where(x < 0.0, -1.0, 1.0)


def remuestreo_sistematico(
    pesos: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Remuestreo insesgado de baja varianza, recomendado en la seccion 6.1."""
    n = pesos.size
    posiciones = (rng.random() + np.arange(n)) / n
    acumulados = np.cumsum(pesos)
    acumulados[-1] = 1.0
    return np.searchsorted(acumulados, posiciones, side="right")


def _pesos_observacion(y_t: float, s: np.ndarray):
    """Devuelve log-media de pesos y pesos normalizados p(y_t|s_t)."""
    log_pesos = (
        -0.5 * ((y_t - s) / SIGMA_OBSERVACION) ** 2
        - np.log(SIGMA_OBSERVACION)
        - LOG_RAIZ_2PI
    )
    maximo = float(np.max(log_pesos))
    pesos_estables = np.exp(log_pesos - maximo)
    suma = float(np.sum(pesos_estables))
    log_media = maximo + np.log(suma) - np.log(s.size)
    return log_media, pesos_estables / suma


def filtro_particulas(
    theta: np.ndarray,
    y: np.ndarray,
    n_particulas: int,
    rng: np.random.Generator,
    guardar_trayectoria: bool = False,
):
    """Estima log p(y|theta) con el filtro bootstrap del Algoritmo 2.

    ``obs.nc`` contiene y_0,...,y_1000. Primero se pondera el estado inicial
    conocido y, desde t=1, se remuestrea, propaga y pondera. Esta es la misma
    recursion del Algoritmo 2 con el indice desplazado para incluir y_0.

    Si ``guardar_trayectoria`` es verdadero, ademas de la log-verosimilitud se
    devuelve una trayectoria completa de estados ``(x^1_{0:T}, x^2_{0:T})``,
    donde x^1_t = s(t) es el desplazamiento y x^2_t = sdot(t) la velocidad,
    con la notacion de la ecuacion (42). La trayectoria se obtiene eligiendo
    una particula final segun sus pesos y siguiendo hacia atras sus indices
    ancestrales a_t. Es la salida de estados que acompaña a PMH (seccion 5 del
    articulo) y permite aproximar la distribucion de suavizado
    p(x_{0:T} | y_{1:T}) de la Figura 5.
    """
    if n_particulas < 2:
        raise ValueError("El filtro requiere al menos dos particulas.")

    f_c, c_0, k, p = theta
    s = np.full(n_particulas, S_INICIAL)
    velocidad = np.full(n_particulas, VELOCIDAD_INICIAL)
    pesos = np.full(n_particulas, 1.0 / n_particulas)
    log_verosimilitud = 0.0

    if guardar_trayectoria:
        # Historial de particulas y de indices ancestrales a_t (Algoritmo 2).
        historial_s = np.empty((y.size, n_particulas))
        historial_velocidad = np.empty((y.size, n_particulas))
        historial_ancestros = np.zeros((y.size, n_particulas), dtype=np.intp)
        historial_s[0] = s
        historial_velocidad[0] = velocidad

    # Observacion inicial. Todos los estados son iguales, pero se incluye su
    # contribucion para reportar la verosimilitud completa del archivo.
    if np.isfinite(y[0]):
        incremento, pesos = _pesos_observacion(float(y[0]), s)
        log_verosimilitud += incremento

    for t in range(1, y.size):
        # Algoritmo 2, paso de remuestreo.
        ancestros = remuestreo_sistematico(pesos, rng)
        s_anterior = s[ancestros]
        velocidad_anterior = velocidad[ancestros]

        # Ecuaciones (42a)-(42b), Euler hacia adelante.
        s = s_anterior + T_S * velocidad_anterior
        fuerza = (
            -f_c * signo_libbi(velocidad_anterior)
            -c_0 * velocidad_anterior
            -k * signo_libbi(s_anterior) * np.abs(s_anterior) ** p
        )
        velocidad = (
            velocidad_anterior
            + (T_S / MASA) * fuerza
            + rng.normal(0.0, SIGMA_PROCESO, n_particulas)
        )

        if guardar_trayectoria:
            historial_s[t] = s
            historial_velocidad[t] = velocidad
            historial_ancestros[t] = ancestros

        # Algoritmo 2, ponderacion. Si falta y_t, la distribucion predictiva
        # pasa sin actualizacion y los pesos quedan uniformes.
        if np.isfinite(y[t]):
            incremento, pesos = _pesos_observacion(float(y[t]), s)
            log_verosimilitud += incremento
        else:
            pesos.fill(1.0 / n_particulas)

    if not guardar_trayectoria:
        return float(log_verosimilitud)

    # Se elige una particula al tiempo final con probabilidad igual a su peso
    # normalizado y se reconstruye su linaje hacia atras con a_t.
    trayectoria = np.empty((2, y.size))
    indice = int(rng.choice(n_particulas, p=pesos))
    for t in range(y.size - 1, -1, -1):
        trayectoria[0, t] = historial_s[t, indice]
        trayectoria[1, t] = historial_velocidad[t, indice]
        if t > 0:
            indice = int(historial_ancestros[t, indice])

    return float(log_verosimilitud), trayectoria


# ---------------------------------------------------------------------------
# Algoritmo 3: particle Metropolis-Hastings
# ---------------------------------------------------------------------------
def pmh(
    y: np.ndarray,
    n_iteraciones: int,
    n_particulas: int,
    semilla: int = 0,
    theta_inicial: np.ndarray | None = None,
    adelgazar_estados: int = 10,
    verbose: bool = True,
):
    """Genera una cadena de vectores theta y trayectorias de estados con PMH.

    Cada iteracion conserva, junto con theta y la verosimilitud estimada, una
    trayectoria x_{0:T} extraida del mismo filtro de particulas. Si la
    propuesta se acepta, se aceptan conjuntamente theta' y su trayectoria; si
    se rechaza, se repite la trayectoria anterior. Las trayectorias se guardan
    cada ``adelgazar_estados`` iteraciones para limitar el tamaño del archivo.
    """
    if n_iteraciones < 2:
        raise ValueError("PMH requiere al menos dos iteraciones.")
    if adelgazar_estados < 1:
        raise ValueError("adelgazar_estados debe ser al menos 1.")

    rng = np.random.default_rng(semilla)
    theta = (
        muestrear_priori(rng)
        if theta_inicial is None
        else np.asarray(theta_inicial, dtype=float).copy()
    )

    if theta.shape != (4,) or not np.isfinite(log_priori(theta)):
        raise ValueError("theta_inicial debe ser [f_c, c_0, k, p] valido.")
    if np.any(theta < LIMITE_INFERIOR) or np.any(theta > LIMITE_SUPERIOR):
        raise ValueError("theta_inicial esta fuera de los limites de propuesta.")

    log_prior = log_priori(theta)
    log_lik, trayectoria = filtro_particulas(
        theta, y, n_particulas, rng, guardar_trayectoria=True
    )

    cadena = np.empty((n_iteraciones, 4))
    log_prioris = np.empty(n_iteraciones)
    log_verosimilitudes = np.empty(n_iteraciones)
    aceptada = np.zeros(n_iteraciones, dtype=np.int8)

    # Trayectorias guardadas: iteraciones 0, k, 2k, ... con k=adelgazar_estados.
    iteraciones_estado = np.arange(0, n_iteraciones, adelgazar_estados)
    estados = np.empty((iteraciones_estado.size, 2, y.size), dtype=np.float32)
    siguiente_estado = 0

    cadena[0] = theta
    log_prioris[0] = log_prior
    log_verosimilitudes[0] = log_lik
    estados[siguiente_estado] = trayectoria
    siguiente_estado += 1

    inicio = perf_counter()
    intervalo = max(1, n_iteraciones // 20)

    for i in range(1, n_iteraciones):
        propuesta, log_cociente_q = proponer_theta(theta, rng)
        log_prior_propuesta = log_priori(propuesta)
        log_lik_propuesta, trayectoria_propuesta = filtro_particulas(
            propuesta, y, n_particulas, rng, guardar_trayectoria=True
        )

        # Ecuacion (33): verosimilitud estimada, prior y cociente de propuestas.
        log_alpha = (
            log_lik_propuesta
            + log_prior_propuesta
            - log_lik
            - log_prior
            + log_cociente_q
        )

        if np.log(rng.random()) < min(0.0, log_alpha):
            theta = propuesta
            log_prior = log_prior_propuesta
            log_lik = log_lik_propuesta
            trayectoria = trayectoria_propuesta
            aceptada[i] = 1

        # En un rechazo se conservan conjuntamente theta, la estimacion de
        # verosimilitud y la trayectoria anteriores, como exige el algoritmo
        # pseudo-marginal.
        cadena[i] = theta
        log_prioris[i] = log_prior
        log_verosimilitudes[i] = log_lik
        if i % adelgazar_estados == 0:
            estados[siguiente_estado] = trayectoria
            siguiente_estado += 1

        if verbose and (i % intervalo == 0 or i == n_iteraciones - 1):
            tasa = aceptada[1:i + 1].mean()
            minutos = (perf_counter() - inicio) / 60.0
            print(
                f"  iter {i:5d}/{n_iteraciones - 1} | "
                f"aceptacion {tasa:6.2%} | {minutos:6.2f} min | "
                f"theta=[{theta[0]:.4f}, {theta[1]:.4f}, "
                f"{theta[2]:.4f}, {theta[3]:.4f}]"
            )

    tasa_aceptacion = float(aceptada[1:].mean())
    minutos_totales = (perf_counter() - inicio) / 60.0
    return {
        "cadena": cadena,
        "log_verosimilitudes": log_verosimilitudes,
        "log_prioris": log_prioris,
        "aceptada": aceptada,
        "tasa_aceptacion": tasa_aceptacion,
        "estados": estados,
        "iteraciones_estado": iteraciones_estado,
        "adelgazar_estados": adelgazar_estados,
        "minutos": minutos_totales,
    }


# ---------------------------------------------------------------------------
# Salida NetCDF reproducible
# ---------------------------------------------------------------------------
def guardar_posterior(
    ruta: str,
    resultado: dict,
    n_particulas: int,
    semilla: int,
    theta_inicial,
) -> str:
    """Guarda muestras, trayectorias de estados y metadatos de la corrida."""
    carpeta = os.path.dirname(ruta)
    if carpeta:
        os.makedirs(carpeta, exist_ok=True)

    cadena = resultado["cadena"]
    estados = resultado["estados"]

    with Dataset(ruta, "w", format="NETCDF4") as ds:
        # Cadena de parametros theta = (f_c, c_0, k, p).
        ds.createDimension("muestra", cadena.shape[0])
        for j, nombre in enumerate(PARAMETROS):
            variable = ds.createVariable(nombre, "f8", ("muestra",))
            variable[:] = cadena[:, j]

        ds.createVariable("loglikelihood", "f8", ("muestra",))[:] = (
            resultado["log_verosimilitudes"]
        )
        ds.createVariable("logprior", "f8", ("muestra",))[:] = (
            resultado["log_prioris"]
        )
        ds.createVariable("aceptada", "i1", ("muestra",))[:] = (
            resultado["aceptada"]
        )

        # Trayectorias de estados, notacion de la ecuacion (42):
        # x1 = x^1_t = s(t) (desplazamiento), x2 = x^2_t = sdot(t) (velocidad).
        ds.createDimension("muestra_estado", estados.shape[0])
        ds.createDimension("tiempo", estados.shape[2])

        iteracion = ds.createVariable("iteracion_estado", "i4", ("muestra_estado",))
        iteracion[:] = resultado["iteraciones_estado"]
        iteracion.long_name = "iteracion PMH de la que procede cada trayectoria"

        indice = ds.createVariable("tiempo", "i4", ("tiempo",))
        indice[:] = np.arange(estados.shape[2])
        indice.long_name = "indice temporal t; tiempo fisico = t * T_s"

        x1 = ds.createVariable(
            "x1", "f4", ("muestra_estado", "tiempo"), zlib=True, complevel=4
        )
        x1[:] = estados[:, 0, :]
        x1.long_name = "x^1_t = s(t), desplazamiento"

        x2 = ds.createVariable(
            "x2", "f4", ("muestra_estado", "tiempo"), zlib=True, complevel=4
        )
        x2[:] = estados[:, 1, :]
        x2.long_name = "x^2_t = sdot(t), velocidad"

        ds.metodo = "Particle Metropolis-Hastings; Algoritmos 2 y 3"
        ds.modelo = "Damper.bi; masa=2; sigma_v=0.01; sigma_y=0.1"
        ds.n_particulas = n_particulas
        ds.n_iteraciones = cadena.shape[0]
        # int64 explicito: la semilla puede superar el rango de 32 bits.
        ds.semilla = np.int64(semilla)
        ds.tasa_aceptacion = resultado["tasa_aceptacion"]
        ds.minutos_ejecucion = resultado["minutos"]
        ds.adelgazar_estados = int(resultado["adelgazar_estados"])
        ds.theta_inicial = (
            "muestreado de las prioris" if theta_inicial is None
            else ",".join(f"{v:g}" for v in theta_inicial)
        )
        ds.orden_parametros = ",".join(PARAMETROS)

    return ruta


def imprimir_resumen(cadena: np.ndarray, verdaderos: dict[str, float]):
    """Imprime un resumen descriptivo después de 20 % de calentamiento."""
    quema = cadena.shape[0] // 5
    print(f"\nResumen posterior provisional (quema = {quema}):")
    print(
        f"{'param':>6} {'verdadero':>10} {'media':>10} {'desv':>10} "
        f"{'IC 2.5%':>10} {'IC 97.5%':>10}"
    )
    for j, nombre in enumerate(PARAMETROS):
        valores = cadena[quema:, j]
        print(
            f"{nombre:>6} {verdaderos.get(nombre, np.nan):>10.4f} "
            f"{valores.mean():>10.4f} {valores.std(ddof=1):>10.4f} "
            f"{np.percentile(valores, 2.5):>10.4f} "
            f"{np.percentile(valores, 97.5):>10.4f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="PMH del ejemplo Damper de Schon et al. (2018)."
    )
    parser.add_argument(
        "--iter", type=int, default=10_000,
        help="iteraciones PMH; el articulo usa 10000",
    )
    parser.add_argument(
        "--part", type=int, default=256,
        help="particulas del filtro; el articulo usa 256",
    )
    parser.add_argument("--semilla", type=int, default=0)
    parser.add_argument(
        "--inicio", type=float, nargs=4,
        metavar=("F_C", "C_0", "K", "P"),
        help="theta inicial opcional, en el orden f_c c_0 k p",
    )
    parser.add_argument(
        "--adelgazar-estados", type=int, default=10,
        help="guardar la trayectoria de estados cada N iteraciones (def. 10)",
    )
    parser.add_argument(
        "--salida",
        default=os.path.join(CARPETA_SALIDA, "posterior_python_corregido.nc"),
        help="archivo NetCDF de salida",
    )
    args = parser.parse_args()

    print("Cargando observaciones simuladas del articulo...")
    _, y, verdaderos = cargar_observaciones()
    print(f"  observaciones: {y.size}")
    print("  parametros verdaderos:", verdaderos)
    print(
        f"\nEjecutando PMH: {args.iter} iteraciones, "
        f"{args.part} particulas, semilla {args.semilla}"
    )

    resultado = pmh(
        y=y,
        n_iteraciones=args.iter,
        n_particulas=args.part,
        semilla=args.semilla,
        theta_inicial=args.inicio,
        adelgazar_estados=args.adelgazar_estados,
    )

    ruta = guardar_posterior(
        ruta=args.salida,
        resultado=resultado,
        n_particulas=args.part,
        semilla=args.semilla,
        theta_inicial=args.inicio,
    )

    print(f"\nTasa de aceptacion final: {resultado['tasa_aceptacion']:.2%}")
    print(f"Tiempo de ejecucion: {resultado['minutos']:.2f} minutos")
    print(
        f"Trayectorias de estados guardadas: {resultado['estados'].shape[0]} "
        f"(una cada {args.adelgazar_estados} iteraciones)"
    )
    imprimir_resumen(resultado["cadena"], verdaderos)
    print(f"\nPosterior guardado en: {ruta}")


if __name__ == "__main__":
    main()
