@echo off
REM Script para ejecutar el entrenamiento distribuido en Windows 10
REM Ejecuta el servidor y 4 workers en paralelo

echo ============================================
echo   Entrenamiento Distribuido - Version select()
echo ============================================
echo.

REM Iniciar el servidor en una nueva ventana
echo Iniciando Server...
start "Server - select()" cmd /k python server_select.py

REM Esperar un momento para que el servidor se inicie
timeout /t 5 /nobreak > nul

REM Iniciar los 4 workers en ventanas separadas
echo Iniciando Workers...
start "Worker 0" cmd /k python worker_select.py
timeout /t 1 /nobreak > nul
start "Worker 1" cmd /k python worker_select.py
timeout /t 1 /nobreak > nul
start "Worker 2" cmd /k python worker_select.py
timeout /t 1 /nobreak > nul
start "Worker 3" cmd /k python worker_select.py

echo.
echo Todos los procesos iniciados.
echo El servidor mostrara el progreso del entrenamiento.
echo.
pause
