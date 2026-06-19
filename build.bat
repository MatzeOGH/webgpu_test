@echo off
setlocal enabledelayedexpansion
cd /D "%~dp0"

for %%a in (%*) do set "%%a=1"
