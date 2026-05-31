@echo off
setlocal

REM One-click Windows entry point for collaborators.
REM It delegates to scripts\setup_env.ps1 so that the real logic is versioned
REM in a readable PowerShell script.

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup_env.ps1" %*
exit /b %ERRORLEVEL%
