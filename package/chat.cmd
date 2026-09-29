@echo off
setlocal
set "L0LLM_MODELS=%~dp0models"
if "%~1"=="" (
  "%~dp0app\l0llm.exe" chat --model "%~dp0models\Qwen2.5-0.5B-Instruct"
) else (
  "%~dp0app\l0llm.exe" chat --model "%~dp0models\%~1"
)
