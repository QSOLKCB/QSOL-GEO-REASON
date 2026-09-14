@echo off
setlocal EnableExtensions DisableDelayedExpansion

rem Windows-native installed qsol-geo-capture trust boundary.  Python starts
rem isolated/no-site/no-bytecode before any QSOL module or editable path import.
set "qsol_python=%~dp0python.exe"
if not exist "%qsol_python%" set "qsol_python=%~dp0..\python.exe"
if not exist "%qsol_python%" (
  echo qsol-geo-capture: installed command cannot locate its Python interpreter 1>&2
  exit /b 126
)

"%qsol_python%" -I -S -B "%~dp0qsol-geo-capture-windows.py" "%~f0" %*
exit /b %ERRORLEVEL%
