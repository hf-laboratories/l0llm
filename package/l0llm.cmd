@echo off
setlocal
set "L0LLM_MODELS=%~dp0models"
"%~dp0app\l0llm.exe" %*
