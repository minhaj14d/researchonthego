@echo off
setlocal EnableExtensions
rem LaTeX Workshop: build without requiring a .tex file to be focused (bypasses root-file detection).
set "PATH=C:\Strawberry\perl\bin;C:\Program Files\MiKTeX\miktex\bin\x64;%PATH%"
set "BIBINPUTS=%~dp0latex;%BIBINPUTS%"
cd /d "%~dp0latex"
"C:\Program Files\MiKTeX\miktex\bin\x64\latexmk.exe" -pdf -g -f -synctex=1 -interaction=nonstopmode -file-line-error -pvc- "main.tex"
exit /b %ERRORLEVEL%
