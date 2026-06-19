@echo off
setlocal
cd /D "%~dp0"

:: Build RenderGraph.cpp standalone (smoke-test driver is #included at the end) and run main().
where cl >nul 2>&1 || call "%~dp0msvcsetup.bat"

cl /nologo /std:c++20 /EHsc ^
   /I build-win\_deps\dawn_native-src\include ^
   src\RenderGraph.cpp ^
   /Fo:build-win\RenderGraph.obj /Fe:build-win\rg.exe

if errorlevel 1 (
    echo RenderGraph build FAILED.
    exit /b 1
)
echo RenderGraph build OK. Running:
build-win\rg.exe
