@echo off
setlocal enabledelayedexpansion

set NANITE_BUILDER=build-win\Release\nanite_builder.exe
set INPUT_MESH=assets\Untitled.gltf
set OUTPUT_MESH=assets\n.mesh

if not exist "%NANITE_BUILDER%" (
    echo Error: %NANITE_BUILDER% not found. Run build_win.bat first.
    exit /b 1
)

if not exist "%INPUT_MESH%" (
    echo Error: %INPUT_MESH% not found.
    exit /b 1
)

echo Building mesh: %INPUT_MESH% -^> %OUTPUT_MESH%
"%NANITE_BUILDER%" "%INPUT_MESH%" "%OUTPUT_MESH%"

if errorlevel 1 (
    echo Error: Mesh build failed.
    exit /b 1
)

echo Mesh build complete: %OUTPUT_MESH%
exit /b 0
