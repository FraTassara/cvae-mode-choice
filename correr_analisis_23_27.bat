@echo off
setlocal enabledelayedexpansion
if "%~1"=="" (
    echo Falta el dataset. Ejemplo: correr_analisis_23_27.bat sm_centro
    exit /b 1
)
set DS=%~1

echo [1/4] Figuras de distribucion (2.3a)...
python python\scripts\09_distribuciones.py --dataset %DS%
if errorlevel 1 exit /b 1

echo [2/4] Efectos marginales (2.3c)...
Rscript R\08_efectos_marginales.R --dataset %DS%
if errorlevel 1 exit /b 1

echo [3/4] Sensibilidad de elipses: 70/90/95 (2.7)...
for %%C in (0.70 0.90 0.95) do (
    set NIVEL=%%C
    set SUF=!NIVEL:0.=conf!
    python python\scripts\02_sinteticos.py --dataset %DS% --metodo CVAE --confianza %%C
    if errorlevel 1 exit /b 1
    Rscript R\03_estimar.R --dataset %DS% --metodo CVAE_!SUF!
    if errorlevel 1 exit /b 1
)

echo [4/4] Consolidando sensibilidad...
python python\scripts\10_sensibilidad_elipses.py --dataset %DS%
echo Listo: %DS%
endlocal