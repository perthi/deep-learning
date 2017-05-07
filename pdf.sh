#!/bin/bash

if [ $# -ne 1 ]; 
    then echo "Usage $0 [filename]"
else
    pdflatex $1 
    pdf=${1%".tex"}.pdf
   ## echo "PDF = $pdf"
    xpdf -g 1200x800  -z page $pdf &
fi
