TEX := hw

all: HW

clean:
	-rm -f $(TEX).pdf

HW: $(TEX).tex
	latexmk -pdf -pdflatex="pdflatex -interaction=nonstopmode" --shell-escape $(TEX).tex
