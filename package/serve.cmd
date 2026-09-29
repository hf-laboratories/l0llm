@echo off
setlocal
cd /d "%~dp0"

set "MODEL_ARG=%~1"
if "%MODEL_ARG%"=="" (
  "%~dp0l0llm.exe" serve
) else (
  if exist "%~dp0models\%MODEL_ARG%" (
    "%~dp0l0llm.exe" serve --model "%~dp0models\%MODEL_ARG%"
  ) else (
    "%~dp0l0llm.exe" serve --model "%MODEL_ARG%"
  )
)
