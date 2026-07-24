@echo off
REM ==========================================================================
REM  correr_todo.bat - Pipeline completo para UN dataset.
REM
REM  Uso:
REM      correr_todo.bat sm_centro
REM      correr_todo.bat sm_centro --con-busqueda
REM
REM  El segundo argumento activa la busqueda de hiperparametros (paso 7).
REM  Ojo: la busqueda NO escribe en config.yaml. Si la corres, el script se
REM  detiene despues para que copies el resultado a mano y lo relances.
REM
REM  Requiere el entorno virtual activado (.venv\Scripts\activate.bat).
REM ==========================================================================

setlocal
if "%~1"=="" (
    echo Falta el dataset. Ejemplo: correr_todo.bat sm_centro
    exit /b 1
)
set DS=%~1
set BUSQUEDA=%~2

echo.
echo ============================================================
echo   Pipeline completo: %DS%
echo ============================================================

REM ---- Paso 0: verificar entorno y columnas -------------------------------
echo.
echo [0/6] Verificando entorno...
python python\scripts\00_verificar.py
if errorlevel 1 (
    echo.
    echo Fallo la verificacion. Corrige antes de seguir.
    exit /b 1
)

REM ---- Paso 1: normalizar y particionar -----------------------------------
echo.
echo [1/6] Normalizando datos y construyendo particiones...
python python\scripts\01_folds.py --dataset %DS%
if errorlevel 1 exit /b 1

REM ---- Paso 7 (opcional): busqueda de hiperparametros ---------------------
if /i "%BUSQUEDA%"=="--con-busqueda" (
    echo.
    echo [7] Busqueda de hiperparametros...
    python python\scripts\07_hiperparametros.py --dataset %DS%
    echo.
    echo ============================================================
    echo   Copia los valores encontrados al bloque cvae.arquitectura
    echo   de config.yaml, dataset '%DS%', y vuelve a lanzar:
    echo       correr_todo.bat %DS%
    echo ============================================================
    exit /b 0
)

REM ---- Paso 2: generar conjuntos de entrenamiento -------------------------
echo.
echo [2/6] Generando conjuntos de entrenamiento (todos los metodos)...
python python\scripts\02_sinteticos.py --dataset %DS%
if errorlevel 1 exit /b 1

REM ---- Verificacion del contrato ------------------------------------------
echo.
echo [2b] Verificando invariantes del contrato...
python python\scripts\tests_contrato.py --dataset %DS%
if errorlevel 1 (
    echo.
    echo El contrato fallo. NO sigas: los resultados serian invalidos.
    exit /b 1
)

REM ---- Paso 3: estimar en R -----------------------------------------------
echo.
echo [3/6] Estimando modelos con Apollo (esto tarda)...
Rscript R\03_estimar.R --dataset %DS%
if errorlevel 1 exit /b 1

REM ---- Paso 4 y 5: tablas y Excel -----------------------------------------
echo.
echo [4/6] Armando tablas...
python python\scripts\04_reporte.py --dataset %DS%
if errorlevel 1 exit /b 1

echo.
echo [5/6] Exportando a Excel...
python python\scripts\05_excel.py --dataset %DS%
if errorlevel 1 exit /b 1

REM ---- Paso 6 y 8: diagnosticos -------------------------------------------
echo.
echo [6/6] Diagnosticos de datos sinteticos...
python python\scripts\06_diagnosticos.py --dataset %DS%

echo.
echo Efectos marginales...
Rscript R\08_efectos_marginales.R --dataset %DS%

echo.
echo ============================================================
echo   Listo: %DS%
echo   Revisa results\tablas_%DS%.xlsx
echo ============================================================
endlocal
