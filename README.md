# Estimación de parámetros de sistema masa–resorte–amortiguador utilizando Particle Metropolis–Hastings

Este repositorio reproduce en Python el ejemplo numérico del sistema no
lineal presentado en:

> T. B. Schön, A. Svensson, L. Murray y F. Lindsten (2018),
> *Probabilistic learning of nonlinear dynamical systems using sequential
> Monte Carlo*, Mechanical Systems and Signal Processing, 104, 866–883.

Enlaces externos:
- [Página del artículo en arXiv](https://arxiv.org/abs/1703.02419)
- [Artículo completo en PDF](https://arxiv.org/pdf/1703.02419)

El repositorio conserva el material original en LibBi y contiene una
reimplementación en Python del modelo, el filtro de partículas bootstrap y el
algoritmo Particle Metropolis–Hastings (PMH).

## Estructura del repositorio

```text
.
├── 1_original_articulo/
│   ├── Damper.bi              # modelo original en LibBi
│   ├── posterior.conf         # 10 000 muestras y 256 partículas
│   ├── config.conf            # configuración de hilos de LibBi
│   ├── run.sh                 # ejecución original
│   ├── data/obs.nc            # datos simulados publicados
│   ├── results/posterior.nc   # salida original sin muestras utilizables
│   └── README.md              # documentación del material original
├── 2_trabajo_python/
│   ├── pmh_damper.py          # modelo, filtro bootstrap y PMH
│   ├── ver_resultados.py      # figuras y resumen posterior
│   ├── requirements.txt       # dependencias de Python
│   ├── data/obs.nc            # copia de las observaciones
│   ├── resultados_python/     # cadenas MCMC en NetCDF
│   ├── figuras/               # gráficos y resumen CSV
│   └── README.md              # documentación técnica de Python
└── README.md
```

## Procedencia de los datos
Los datos contenidos en `obs.nc` no corresponden a mediciones experimentales
reales. Fueron generados mediante una simulación del sistema no lineal
masa–resorte–amortiguador descrito en la sección 7 del artículo. Los parámetros
verdaderos utilizados durante la simulación también se encuentran almacenados
en el archivo para facilitar la comparación con las estimaciones posteriores.

El 23 de octubre de 2017, los autores anunciaron la aceptación del artículo en
*Mechanical Systems and Signal Processing* y publicaron los enlaces al
manuscrito, la versión de arXiv y el código necesario para reproducir el ejemplo.
La entrada correspondiente puede consultarse en la
[página de resultados de investigación de Thomas B. Schön](https://acortar.link/2p4LCt).
Dentro de esa entrada aparecen los enlaces identificados como **[pdf]**,
**[arXiv]** y **[code]**. El archivo `obs.nc` utilizado en este repositorio
procede del paquete distribuido mediante el enlace **[code]**.

## Requisitos

- Python. La ejecución de referencia se probó con **Python 3.14.5**.
- Espacio para un entorno virtual y las dependencias científicas.
- Aproximadamente 10 minutos para la configuración de referencia; el tiempo
  depende del procesador.

No se necesita LibBi para ejecutar la versión Python.

> No suba `.venv/`, `__pycache__/` ni archivos temporales a GitHub. Cada persona
> debe crear su propio entorno virtual con los comandos siguientes.

## Instalación en Windows PowerShell

Ejecute estos comandos desde la raíz del repositorio:

```powershell
# 1. Crear el entorno virtual
py -m venv .venv

# 2. Actualizar pip e instalar las dependencias
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r .\2_trabajo_python\requirements.txt
```

Se usa directamente el ejecutable del entorno. Por eso no es necesario cambiar
la política de ejecución de PowerShell para activar `.venv`.

## Instalación en Linux o macOS

Ejecute estos comandos desde la raíz del repositorio:

```bash
# 1. Crear y activar el entorno virtual
python3 -m venv .venv
source .venv/bin/activate

# 2. Actualizar pip e instalar las dependencias
python -m pip install --upgrade pip
python -m pip install -r 2_trabajo_python/requirements.txt
```

## Prueba rápida

Esta ejecución solo verifica que la instalación funciona. No produce una cadena
suficiente para hacer inferencia.

### Windows PowerShell

```powershell
.\.venv\Scripts\python.exe .\2_trabajo_python\pmh_damper.py --iter 200 --part 64 --semilla 1 --salida resultados_python/posterior_prueba.nc
```

### Linux o macOS

```bash
python 2_trabajo_python/pmh_damper.py --iter 200 --part 64 --semilla 1 --salida resultados_python/posterior_prueba.nc
```

## Ejecutar la inferencia

Los dos argumentos principales son:

- `--iter`: cantidad de iteraciones de la cadena PMH. Cada iteración guarda un
  vector `theta = (f_c, c_0, k, p)`; las propuestas rechazadas repiten el vector
  anterior.
- `--part`: cantidad de partículas internas usadas para estimar la
  verosimilitud en cada iteración. No son muestras adicionales de `theta`.

Puede elegir ambos valores. Por ejemplo, en Windows PowerShell:

```powershell
$ITERACIONES = 10000
$PARTICULAS = 256

.\.venv\Scripts\python.exe .\2_trabajo_python\pmh_damper.py `
  --iter $ITERACIONES `
  --part $PARTICULAS `
  --semilla 2026
```

En Linux o macOS:

```bash
ITERACIONES=10000
PARTICULAS=256

python 2_trabajo_python/pmh_damper.py \
  --iter "$ITERACIONES" \
  --part "$PARTICULAS" \
  --semilla 2026
```

La salida predeterminada es:

```text
2_trabajo_python/resultados_python/posterior_python_corregido.nc
```

El costo computacional crece aproximadamente con el producto
`iteraciones × partículas`. Duplicar cualquiera de los dos valores casi duplica
el tiempo de ejecución.

### Valores recomendados

| Propósito | Iteraciones | Partículas | Uso |
|---|---:|---:|---|
| Verificar instalación | 200 | 64 | Solo prueba técnica |
| Exploración preliminar | 2 000 | 128 | Revisar código y trazas |
| Reproducir la configuración publicada | **10 000** | **256** | Resultado principal |
| Diagnóstico más exigente | 50 000 o varias cadenas | 256 o más | Evaluar convergencia y ESS |

La recomendación principal es **10 000 iteraciones y 256 partículas**, porque
son los valores del comando LibBi publicado y los usados en la última ejecución
de este proyecto.

## Última ejecución realizada

La última corrida completa utilizó:

```text
Iteraciones:       10 000
Partículas:        256
Semilla:           2026
Punto inicial:     parámetros verdaderos del ejemplo simulado
Tiempo aproximado: 9.09 minutos
Entorno probado:   Windows 11, Python 3.14.5
Aceptación:        37.77 %
```

Comando equivalente en Windows:

```powershell
.\.venv\Scripts\python.exe .\2_trabajo_python\pmh_damper.py `
  --iter 10000 `
  --part 256 `
  --semilla 2026 `
  --inicio 0.011413877 0.712997154 2.155382464 0.577213338
```

El punto inicial verdadero solo se usó porque este es un ejemplo simulado cuyos
valores se conocen. Para una ejecución ordinaria, omita `--inicio`; el programa
obtendrá un punto inicial a partir de las prioris.

## Generar las figuras

Después de generar el posterior, ejecute:

### Windows PowerShell

```powershell
.\.venv\Scripts\python.exe .\2_trabajo_python\ver_resultados.py
```

### Linux o macOS

```bash
python 2_trabajo_python/ver_resultados.py
```

Se producen versiones PNG de 300 dpi para vista previa y PDF vectoriales para
informes, presentaciones e impresión:

```text
2_trabajo_python/figuras/01_datos_observados.png
2_trabajo_python/figuras/01_datos_observados.pdf
2_trabajo_python/figuras/02_posteriores.png
2_trabajo_python/figuras/02_posteriores.pdf
2_trabajo_python/figuras/03_trazas.png
2_trabajo_python/figuras/03_trazas.pdf
2_trabajo_python/figuras/resumen_posterior.csv
```

`02_posteriores.png` muestra para cada parámetro:

- histograma posterior en azul;
- densidad prior en negro;
- valor verdadero en rojo.

El script descarta el 20 % inicial para el histograma y el resumen. Esta
exclusión se denomina *burn-in* o calentamiento. La gráfica de trazas conserva
todas las iteraciones para que el usuario pueda evaluar visualmente ese periodo.

## Parámetros opcionales del programa

```text
--iter N                     número de iteraciones PMH
--part N                     número de partículas del filtro
--semilla N                  semilla aleatoria
--inicio FC C0 K P           vector inicial opcional
--salida RUTA                archivo NetCDF de salida
```

Para consultar la ayuda completa:

```powershell
.\.venv\Scripts\python.exe .\2_trabajo_python\pmh_damper.py --help
```

## Observaciones científicas importantes

- El cuerpo del artículo declara `m=8`, pero el apéndice `Damper.bi` declara
  `m=2`. La implementación Python usa `m=2` porque coincide con el código
  ejecutable y reproduce mejor `obs.nc`.
- Las 10 000 iteraciones MCMC están autocorrelacionadas. No equivalen a 10 000
  muestras independientes. Se deben revisar las trazas y el tamaño efectivo de
  muestra antes de afirmar convergencia.

## Documentación adicional

- [`1_original_articulo/README.md`](1_original_articulo/README.md): material y
  ejecución original en LibBi.
- [`2_trabajo_python/README.md`](2_trabajo_python/README.md): detalles del
  modelo, los algoritmos, las salidas y la interpretación de resultados.
