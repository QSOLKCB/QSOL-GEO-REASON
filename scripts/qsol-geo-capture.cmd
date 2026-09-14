@echo off
setlocal EnableExtensions DisableDelayedExpansion

rem Windows-native installed qsol-geo-capture trust boundary.  The adjacent
rem interpreter starts isolated/no-site/no-bytecode before any QSOL module or
rem editable-install path is imported.
set "qsol_python=%~dp0python.exe"
if not exist "%qsol_python%" (
  >&2 echo qsol-geo-capture: installed command requires an adjacent python.exe
  exit /b 126
)

"%qsol_python%" -I -S -B "%~dp0qsol-geo-capture-windows.py" "%~f0" %*
exit /b %ERRORLEVEL%
