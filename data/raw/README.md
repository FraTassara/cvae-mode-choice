# Datos crudos

Los CSV originales van acá. `config.yaml` declara el nombre y el separador de
cada uno bajo `datasets:`.

| dataset | archivo | sep | fuente |
|---|---|---|---|
| sm_centro | `sm_database.csv` | `,` | encuesta Santiago Centro |
| lc_centro | `lc_database.csv` | `,` | encuesta Las Condes |
| whalen | `whalen_database.csv` | espacio | Whalen et al. |
| swissmetro | `swissmetro.csv` | `;` | Bierlaire et al., biogeme.epfl.ch |

Si los archivos son demasiado grandes o no se pueden redistribuir, sustituye
esta tabla por el DOI de Zenodo y las instrucciones de descarga. El repo debe
poder correr para cualquiera que siga estas instrucciones.
