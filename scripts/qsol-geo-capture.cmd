@echo off
setlocal EnableExtensions DisableDelayedExpansion

rem Windows-native installed qsol-geo-capture trust boundary. Python starts
rem isolated/no-site/no-bytecode before any QSOL module or editable path import.
set "qsol_python=%~dp0python.exe"
if not exist "%qsol_python%" set "qsol_python=%~dp0..\python.exe"
if exist "%qsol_python%" goto qsol_direct_python

rem A Windows --user install places scripts beneath <userbase>\PythonXY\Scripts
rem while python.exe remains in the registered base installation. Use the fixed
rem Python Launcher trust boundary in that layout rather than requiring an
rem adjacent interpreter or consulting PATH.
for %%I in ("%~dp0..") do set "qsol_user_tag=%%~nxI"
set "qsol_selector="
if /I "%qsol_user_tag%"=="Python311" set "qsol_selector=-3.11"
if /I "%qsol_user_tag%"=="Python312" set "qsol_selector=-3.12"
if /I "%qsol_user_tag%"=="Python313" set "qsol_selector=-3.13"
if not defined qsol_selector (
  echo qsol-geo-capture: cannot derive a supported Python version from the Windows user-site Scripts directory 1>&2
  exit /b 126
)
set "qsol_launcher=C:\Windows\py.exe"
if not exist "%qsol_launcher%" set "qsol_launcher=%LOCALAPPDATA%\Programs\Python\Launcher\py.exe"
if not exist "%qsol_launcher%" (
  echo qsol-geo-capture: Windows user-site install requires the Python Launcher 1>&2
  exit /b 126
)
"%qsol_launcher%" %qsol_selector% -I -S -B "%~dp0qsol-geo-capture-windows.py" "%~f0" %*
exit /b %ERRORLEVEL%

:qsol_direct_python
"%qsol_python%" -I -S -B "%~dp0qsol-geo-capture-windows.py" "%~f0" %*
exit /b %ERRORLEVEL%
