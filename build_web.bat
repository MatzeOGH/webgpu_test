@echo off
setlocal

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

cmake --build "%BUILD_DIR%"
if errorlevel 1 goto :error

echo.
echo Build successful! Output: %BUILD_DIR%\app.html
echo.
echo To serve:
echo   python -m http.server 8080 --directory %BUILD_DIR%
goto :eof

:error
echo.
echo Build failed.
exit /b 1
