@ECHO OFF
REM audiveris.bat
setlocal
java -Xmx256M -jar dist/audiveris-3.3.jar %1 %2 %3 %4 %5 %6 %7 %8 %9
endlocal
        