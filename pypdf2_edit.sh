FILE="venv\Lib\site-packages\PyPDF2\pdf.py"
NUM_LINES=$(wc -l "$FILE")
[[ $NUM_LINES -gtr 3005 ]] || {exit 1; }

sed -i "2620a            elif operator == b_(\"Td\"):" $FILE
sed -i "2621a                if operands[0] < 0:" $FILE
sed -i "2622a                    text += \"\\n\"" $FILE
