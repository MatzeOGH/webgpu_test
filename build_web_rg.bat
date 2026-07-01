@echo off
setlocal

:: Build the RenderGraph sample as a WebAssembly page (Emscripten). Output:
::   build-web\rendergraph\index.html   (served by GitHub Pages at /rendergraph)
:: The other web target (app, the clustered renderer) is NOT built here -- only --target rg.

set EMSDK=C:\Users\huerbe\Documents\emsdk
set EMSCRIPTEN=%EMSDK%\upstream\emscripten
set BUILD_DIR=build-web
set CONFIG=%1
if "%CONFIG%"=="" set CONFIG=Release

if not exist "%EMSCRIPTEN%\emcmake.bat" (
    echo ERROR: emcmake not found at %EMSCRIPTEN%
    echo Check that emsdk is installed at %EMSDK%
    exit /b 1
)

call "%EMSCRIPTEN%\emcmake.bat" cmake -B "%BUILD_DIR%" -S . -DCMAKE_BUILD_TYPE=%CONFIG%
if errorlevel 1 goto :error

cmake --build "%BUILD_DIR%" --target rg
if errorlevel 1 goto :error

echo.
echo Build successful! Output: %BUILD_DIR%\rendergraph\index.html
echo.
echo To serve:
echo   python -m http.server 8080 --directory %BUILD_DIR%
echo Then open: http://localhost:8080/rendergraph/
goto :eof

:error
echo.
echo Build failed.
exit /b 1
