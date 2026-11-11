@echo off
setlocal
where code >nul 2>nul
if errorlevel 1 (
  echo VS Code command "code" was not found on PATH.
  echo Open Visual Studio Code and choose File ^> Open Workspace from File.
  pause
  exit /b 1
)
code "%~dp0EHBA_3D_Radiomics.code-workspace"
