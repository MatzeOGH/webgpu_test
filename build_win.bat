@echo off
setlocal

set BUILD_DIR=build-win
set CONFIG=%1
if "%CONFIG%"=="" set CONFIG=Release

cmake -B "%BUILD_DIR%" -S .
if errorlevel 1 goto :error

cmake --build "%BUILD_DIR%" --config %CONFIG%
if errorlevel 1 goto :error

echo.
echo Build successful! Output: %BUILD_DIR%\%CONFIG%\app.exe
echo.
echo To run:
echo   %BUILD_DIR%\%CONFIG%\app.exe
goto :eof

:error
echo.
echo Build failed.
exit /b 1
