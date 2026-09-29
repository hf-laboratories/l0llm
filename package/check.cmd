@echo off
setlocal
set "L0LLM_MODELS=%~dp0models"
echo === GPU and models ===
"%~dp0app\l0llm.exe" info
echo.
echo === Smoke test (Qwen2.5-0.5B-Instruct) ===
"%~dp0app\l0llm.exe" check --model "%~dp0models\Qwen2.5-0.5B-Instruct"
echo.
echo === Benchmark ===
"%~dp0app\l0llm.exe" bench --model "%~dp0models\Qwen2.5-0.5B-Instruct"
pause
