@echo off
setlocal
cd /D "%~dp0"

:: Build RenderGraph.cpp standalone (smoke-test driver is #included at the end) and run main().
where cl >nul 2>&1 || call "%~dp0msvcsetup.bat"

cl /nologo /std:c++20 /EHsc /MD ^
   /DIMGUI_IMPL_WEBGPU_BACKEND_DAWN ^
   /I build-win\_deps\dawn_native-src\include ^
   /I build-win\_deps\sdl3-src\include ^
   /I src\extern\imgui ^
   /I src\extern\imgui\backends ^
   src\RenderGraph.cpp ^
   src\extern\imgui\imgui.cpp ^
   src\extern\imgui\imgui_draw.cpp ^
   src\extern\imgui\imgui_tables.cpp ^
   src\extern\imgui\imgui_widgets.cpp ^
   src\extern\imgui\backends\imgui_impl_sdl3.cpp ^
   src\extern\imgui\backends\imgui_impl_wgpu.cpp ^
   build-win\_deps\dawn_native-src\lib\webgpu_dawn.lib ^
   build-win\_deps\sdl3-src\lib\x64\SDL3.lib ^
   dxguid.lib ^
   /Fo:build-win\ /Fe:build-win\Release\rg.exe ^
   /link /DEFAULTLIB:onecore.lib

if errorlevel 1 (
    echo RenderGraph build FAILED.
    exit /b 1
)
echo RenderGraph build OK. Running:
:: run from Release\ so Dawn finds dxil.dll / dxcompiler.dll / d3dcompiler_47.dll staged there
build-win\Release\rg.exe
