# === === === === === === === === === === === 
# @Author : Liaw
# @Date   : 08.07.2025
#
# Makefile to compile a LaTeX project using
# your own customized Docker image based on
# texlive's official Docker image
# 
# === === === === === === === === === === ===

# Name of the the "main" file to compile
NAME=main

DOCKER_LATEXCOMPILER=liawlatex

all:
	docker run --rm -v "$(shell pwd):/project" -w /project $(DOCKER_LATEXCOMPILER) latexmk -pdf -file-line-error -bibtex $(NAME).tex

clean:
	rm -f *.toc
	rm -f $(NAME).dvi
	rm -f *.log
	rm -f *.aux
	rm -f $(NAME).ps
	rm -f $(NAME).pdf
	rm -f *~
	rm -f *.fls
	rm -f *.fdb_latexmk
	rm -f *.bbl
	rm -f *.blg
	rm -f *.glo
	rm -f *.ist
	rm -f *.acn
	rm -f *.bcf*
	rm -f *.bbl*
	rm -f *.run*
	rm -f *.acr
	rm -f *.alg
	rm -f *.glg
	rm -f *.gls
	rm -f *.out
	rm -f *.glsdefs
