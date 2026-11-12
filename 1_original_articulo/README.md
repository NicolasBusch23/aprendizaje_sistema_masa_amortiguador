# Material original del artículo

Esta carpeta conserva el código LibBi utilizado para el ejemplo del
amortiguador no lineal. El artículo se consulta mediante los enlaces externos
indicados abajo.

> T. B. Schön, A. Svensson, L. Murray y F. Lindsten (2018),
> *Probabilistic learning of nonlinear dynamical systems using sequential
> Monte Carlo*, Mechanical Systems and Signal Processing, 104, 866–883.

Enlaces externos:

- [Página del artículo en arXiv](https://arxiv.org/abs/1703.02419)
- [Artículo completo en PDF](https://arxiv.org/pdf/1703.02419)

La implementación original no fue modificada. La versión ejecutable en Python
está en [`../2_trabajo_python`](../2_trabajo_python/README.md).

## Contenido

| Ruta | Descripción |
|---|---|
| `Damper.bi` | Modelo original en el lenguaje de modelado de LibBi |
| `posterior.conf` | Configuración PMH: 10 000 muestras, 256 partículas y horizonte 1000 |
| `config.conf` | Configuración de cuatro hilos para LibBi |
| `run.sh` | Script que combina las configuraciones y ejecuta LibBi |
| `data/obs.nc` | Datos simulados: 1001 registros, parámetros verdaderos y eje de salida |
| `results/posterior.nc` | Archivo distribuido con el paquete; contiene valores de relleno, no una cadena posterior utilizable |
| `readme.txt` | Nota original de los autores, fechada en febrero de 2017 |

## Modelo de `Damper.bi`

El archivo declara:

- estados: desplazamiento `s` y velocidad `sdot`;
- parámetros: `f_c`, `c_0`, `k` y `p`;
- estado inicial: `s=0.5` y `sdot=0`;
- paso de muestreo: `T_s=0.1`;
- masa: `m=2`;
- ruido de proceso: `gaussian(0, 0.01)`;
- observación: `gaussian(s, 0.1)`;
- prioris Gamma para `f_c`, `c_0` y `k`, y Uniforme para `p`;
- propuestas gaussianas truncadas para PMH.

## Configuración original

`posterior.conf` equivale al comando publicado:

```sh
libbi sample \
  --target posterior \
  --model-file Damper.bi \
  --nsamples 10000 \
  --nparticles 256 \
  --end-time 1000 \
  --sampler mh \
  --obs-file data/obs.nc \
  --output-file results/posterior.nc
```

`run.sh` carga además `config.conf`, que solicita cuatro hilos:

```sh
libbi sample @config.conf @posterior.conf
```

## Ejecución con LibBi

Esta ruta requiere una instalación funcional de LibBi y un entorno de shell
compatible con `sh`. Desde esta carpeta:

```sh
sh run.sh
```

La instalación de LibBi no forma parte de este repositorio. Consulte el sitio y
la documentación de LibBi indicados por los autores en `readme.txt`.

## Limitación del resultado incluido

El archivo `results/posterior.nc` incluido en el material original no contiene
las 10 000 muestras: sus variables tienen valores de relleno. No debe utilizarse
para elaborar histogramas ni resúmenes.

La cadena generada y verificada en este proyecto se encuentra en:

```text
../2_trabajo_python/resultados_python/posterior_python_corregido.nc
```

## Diferencia entre el texto y el apéndice

La sección 7 del artículo escribe `m=8`, mientras que el apéndice y `Damper.bi`
usan `m=2`. La reproducción Python documenta esta discrepancia y adopta `m=2`
para mantenerse alineada con el código original y los datos proporcionados.
