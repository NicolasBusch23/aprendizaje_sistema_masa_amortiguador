# Implementación Python del ejemplo Damper

Esta carpeta contiene una reimplementación documentada del ejemplo numérico del
artículo de Schön et al. (2018). No ejecuta LibBi: implementa directamente en
Python el mismo modelo de estados y los algoritmos Monte Carlo descritos en el
artículo.

Para instalar el entorno desde cero, consulte primero el
[`README principal`](../README.md).

## Objetivo

Estimar la distribución posterior de

```text
theta = (f_c, c_0, k, p)
```

a partir del desplazamiento observado `y` almacenado en `data/obs.nc`.

- `f_c`: fricción de Coulomb.
- `c_0`: amortiguamiento viscoso.
- `k`: rigidez del resorte.
- `p`: exponente no lineal.

Además, se estiman los estados ocultos del sistema mediante la distribución de
suavizado `p(x_{0:T} | y_{1:T})`, como en la Figura 5 del artículo.

## Modelo implementado

Se conserva la notación de la ecuación (42) del artículo. El estado es
`x_t = (x¹_t, x²_t)`, donde:

- `x¹_t = s(t)`: desplazamiento de la masa;
- `x²_t = ṡ(t)`: velocidad de la masa.

Ambos estados evolucionan mediante Euler hacia adelante y solo se observa el
desplazamiento con ruido:

```text
x¹_{t+1} = x¹_t + T_s · x²_t                                          (42a)

x²_{t+1} = x²_t + (T_s / m) · ( −f_c · signo(x²_t)
                                 − c_0 · x²_t
                                 − k · signo(x¹_t) · |x¹_t|^p ) + v_t   (42b)

y_t      = x¹_t + e_t                                                  (42c)
```

El término `signo(x¹_t) · |x¹_t|^p` reproduce el código `Damper.bi`. La ecuación
impresa escribe `k (x¹_t)^p`, que no está definida para desplazamientos
negativos cuando `p < 1`.

Constantes utilizadas:

| Cantidad | Valor |
|---|---:|
| `T_s` | 0.1 |
| `m` | 2.0 |
| `sigma_v` (ruido de proceso `v_t`) | 0.01 |
| `sigma_y` (ruido de medición `e_t`) | 0.1 |
| `x¹_0 = s(0)` | 0.5 |
| `x²_0 = ṡ(0)` | 0.0 |

La masa sigue el `Damper.bi` del apéndice. El texto del artículo escribe `m=8`,
pero ese valor contradice el código original y no reproduce adecuadamente
`obs.nc`.

## Prioris

```text
f_c ~ Gamma(forma=2, escala=0.01)
c_0 ~ Gamma(forma=2, escala=1)
k   ~ Gamma(forma=4, escala=0.3)
p   ~ Uniforme(0, 1)
```

## Algoritmos

### Filtro de partículas bootstrap

`filtro_particulas()` implementa el Algoritmo 2 y estima
`log p(y | theta)`:

1. Inicializa 256 posibles estados.
2. Evalúa la observación para obtener pesos.
3. Remuestrea los estados según esos pesos.
4. Propaga las partículas mediante la dinámica.
5. Repite el proceso para toda la serie.

Se usa remuestreo sistemático, un método insesgado de baja varianza recomendado
en la sección 6.1 del artículo.

### Particle Metropolis–Hastings

`pmh()` implementa el Algoritmo 3:

1. Propone un nuevo vector `theta` mediante gaussianas truncadas.
2. Ejecuta el filtro de partículas para estimar su verosimilitud.
3. Calcula la probabilidad de aceptación de la ecuación (33), incluido el
   cociente de las densidades de propuesta truncadas.
4. Acepta el candidato o conserva conjuntamente el vector y la estimación de
   verosimilitud anteriores.

### Estimación de los estados

La sección 5 del artículo indica que las trayectorias producidas por los
filtros de partículas que se ejecutan dentro de PMH aproximan la distribución
de suavizado `p(x_{0:T} | y_{1:T})`. Esa es la información que muestra su
Figura 5. Este proyecto la obtiene de la siguiente forma:

1. Durante el filtro bootstrap se guardan, en cada tiempo `t`, las partículas
   `(x¹_t, x²_t)` y los índices ancestrales `a_t` del remuestreo.
2. Al terminar el filtro se elige una partícula final con probabilidad igual a
   su peso normalizado.
3. Se sigue su linaje hacia atrás con `a_t` y se obtiene una trayectoria
   completa `x_{0:T}`.
4. En PMH, esa trayectoria se acepta o se rechaza junto con `theta'`. Si la
   propuesta se rechaza, se repite la trayectoria anterior.
5. Se guarda una trayectoria cada 10 iteraciones (`--adelgazar-estados 10`) para
   que el archivo no supere unos pocos megabytes.

Después del burn-in, las trayectorias guardadas forman una muestra de
`p(x_{0:T} | y_{1:T})` que también incorpora la incertidumbre sobre `theta`.
La media y los percentiles 2.5 % y 97.5 % de esa muestra son los que se grafican
en `04_estados`.

`obs.nc` no contiene los estados verdaderos usados en la simulación. Por eso el
desplazamiento estimado se compara con los datos `y_t` y la velocidad estimada
se muestra sin una referencia.

## Archivos

| Ruta | Descripción |
|---|---|
| `pmh_damper.py` | Modelo, prioris, propuesta truncada, filtro bootstrap, PMH, trayectorias de estados y guardado NetCDF |
| `ver_resultados.py` | Lectura del posterior, figuras y tabla resumen |
| `requirements.txt` | Versiones de las dependencias utilizadas |
| `data/obs.nc` | Copia de los datos simulados del artículo |
| `resultados_python/posterior_python_corregido.nc` | Última cadena completa de 10 000 iteraciones y 1 000 trayectorias de estados `x1`, `x2` |
| `resultados_python/posterior_diagnostico.nc` | Corrida corta usada para validar la implementación |
| `resultados_python/posterior_python.nc` | Corrida antigua de 40 iteraciones; no usar para inferencia |
| `figuras/01_datos_observados.png` y `.pdf` | Serie observada en segundos |
| `figuras/02_posteriores.png` y `.pdf` | Posteriores, prioris y valores verdaderos |
| `figuras/03_trazas.png` y `.pdf` | Trazas MCMC completas |
| `figuras/04_estados.png` y `.pdf` | Desplazamiento `x¹_t` estimado frente a los datos y velocidad `x²_t` estimada |
| `figuras/resumen_posterior.csv` | Media, mediana, desviación e intervalo de cada parámetro |

## Ejecutar PMH desde esta carpeta

Active primero el entorno creado según el README principal. Después puede elegir
las iteraciones y las partículas:

```powershell
python pmh_damper.py --iter 10000 --part 256 --semilla 101112131415
```

Valores disponibles:

```text
--iter N                 longitud de la cadena de parámetros
--part N                 partículas por estimación de verosimilitud
--semilla N              semilla del generador aleatorio
--inicio FC C0 K P       punto inicial opcional
--adelgazar-estados N    guardar una trayectoria de estados cada N iteraciones (def. 10)
--salida RUTA            ruta del NetCDF producido
```

El programa utiliza 10 000 iteraciones y 256 partículas si se omiten `--iter` y
`--part`:

```powershell
python pmh_damper.py
```

### Última corrida

```powershell
python pmh_damper.py `
  --iter 10000 `
  --part 256 `
  --semilla 101112131415 `
  --inicio 0.1 1 3 0.4
```

Esta ejecución tardó cerca de 11.43 minutos en Windows 11 con Python 3.14.5,
obtuvo una tasa de aceptación de 35.04 % y guardó 1 000 trayectorias de
estados. El archivo resultante ocupa unos 6.1 MB. Guardar las trayectorias
añade alrededor de 13 % al tiempo de cada filtro de partículas.

El punto inicial `[0.1, 1, 3, 0.4]` se escogió deliberadamente apartado de los
valores verdaderos (`[0.0114, 0.713, 2.155, 0.577]`) para observar cómo la
cadena se aleja de esa región y converge hacia el modo posterior. Para un uso
ordinario, omita `--inicio`.

## Generar resultados gráficos

Después de ejecutar PMH:

```powershell
python ver_resultados.py
```

Cada gráfico se guarda en dos formatos:

- `.pdf`: salida vectorial recomendada para informes, presentaciones, LaTeX e
  impresión; puede ampliarse sin perder nitidez.
- `.png`: vista previa de 300 dpi, conveniente para GitHub y documentos que no
  acepten gráficos vectoriales.

No se genera EPS porque PDF conserva vectores y transparencias, tiene mejor
soporte actual y suele producir archivos más fáciles de incorporar.

El script prefiere `posterior_python_corregido.nc`. Si ese archivo no existe,
puede leer la corrida antigua para compatibilidad, pero mostrará una advertencia
cuando la cadena sea demasiado corta o tenga pocos valores distintos.

### Figura de datos

`01_datos_observados.png` convierte el índice de `obs.nc` a tiempo físico:

```text
tiempo_en_segundos = indice * 0.1
```

### Figura posterior

`02_posteriores.png` reproduce la estructura de la Figura 4 del artículo:

- histograma posterior azul;
- densidad prior negra;
- línea roja en el valor verdadero.

### Burn-in

Para histogramas y resúmenes se descarta el primer 20 % de la cadena. Con
10 000 iteraciones:

```text
Burn-in:             2 000 iteraciones
Muestras analizadas: 8 000 iteraciones
```

Las muestras no se borran del NetCDF. `03_trazas.png` muestra la cadena completa,
incluido el burn-in, para permitir su inspección. Cada panel tiene una leyenda
que distingue la cadena MCMC (gris) del valor verdadero (línea roja punteada).

### Figura de estados

`04_estados.png` sigue la idea de la Figura 5 del artículo, separada en dos
paneles que comparten el eje de tiempo en segundos:

- **Panel superior:** desplazamiento `x¹_t = s(t)`. Muestra los datos simulados
  `y_t` como puntos negros, la media posterior en azul y el intervalo de 95 %
  sombreado en azul.
- **Panel inferior:** velocidad `x²_t = ṡ(t)`. Muestra la media posterior en
  naranja y el intervalo de 95 % sombreado en naranja. La velocidad no se mide,
  por lo que no hay datos que superponer.

Solo se usan las trayectorias guardadas después del burn-in.

## Resultado de la última corrida

Después del burn-in de 2 000 iteraciones:

| Parámetro | Verdadero | Media | Desviación | IC 95 % |
|---|---:|---:|---:|---:|
| `f_c` | 0.011414 | 0.010988 | 0.007973 | [0.001178, 0.035913] |
| `c_0` | 0.712997 | 0.729189 | 0.061216 | [0.586697, 0.839803] |
| `k` | 2.155382 | 2.069543 | 0.084972 | [1.916553, 2.224688] |
| `p` | 0.577213 | 0.556842 | 0.020217 | [0.518952, 0.595731] |

Para los estados se usan las 800 trayectorias guardadas después del burn-in
(iteraciones 2 000 a 9 990). La media posterior del desplazamiento `x¹_t`
sigue la oscilación amortiguada de los datos `y_t`, y su intervalo de 95 % es
mucho más estrecho que la dispersión de los datos, porque el filtro separa el
ruido de medición `e_t` (desviación 0.1) de la dinámica del sistema. La
velocidad `x²_t` se reconstruye sin haber sido medida, solo a partir del
modelo y de las observaciones de posición.

## Interpretación y convergencia

Las iteraciones MCMC son correlacionadas. La última cadena tiene una aceptación
de 35.04 %, pero sus tamaños efectivos de muestra son bajos, aproximadamente
entre 14 y 25 según el parámetro. Por ello:

- los histogramas reproducen razonablemente la forma publicada;
- los resúmenes deben considerarse aproximaciones Monte Carlo;
- una tasa de aceptación razonable no demuestra convergencia;
- para un análisis formal conviene ejecutar varias cadenas desde puntos iniciales
  dispersos, calcular `R-hat` y aumentar la longitud hasta obtener un ESS mayor.

El artículo no especifica una semilla ni proporciona una cadena posterior
utilizable. Por eso no se espera igualdad muestra por muestra con LibBi, sino
resultados estadísticamente comparables.
