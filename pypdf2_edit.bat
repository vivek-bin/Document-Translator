@ECHO OFF
SET /a line=2620
SET file="venv\Lib\site-packages\PyPDF2\pdf.py"
SET tempfile="venv\Lib\site-packages\PyPDF2\pdf.tmp"

FOR /f "tokens=1* delims=:" %%a IN ('findstr /n "^" %file%') DO SET size=%%a
IF %size% gtr 3005 (GOTO :EOF)

FOR /f "tokens=1* delims=:" %%a IN ('findstr /n "^" %file%') DO IF %%a leq %line% echo(%%b>>%tempfile%

echo(            elif operator == b_("Td"): >> %tempfile%
echo(                if operands[0] ^< 0: >> %tempfile%
echo(                    text += "\n" >> %tempfile%

FOR /f "tokens=1* delims=:" %%a IN ('findstr /n "^" %file%') DO IF %%a gtr %line% echo(%%b>>%tempfile%

DEL /f %file%
TYPE %tempfile% > %file%
DEL /f %tempfile%

GOTO :EOF