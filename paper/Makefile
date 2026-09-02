latex=/usr/bin/latex
dvips=/usr/bin/dvips
dvipdf=/usr/bin/dvipdf
bibtex=/usr/bin/bibtex

prefix=round_trip_knn_nca_sn_draft_popularity_visibility

aux=$(prefix).aux
dvi=$(prefix).dvi
log=$(prefix).log
out=$(prefix).out
tex=$(prefix).tex
pdf=$(prefix).pdf
bbl=$(prefix).bbl
blg=$(prefix).blg
bib=$(prefix).bib

rmlist=\
	$(aux) \
	$(dvi) \
	$(log) \
	$(bbl) \
	$(blg) \
	$(out)

$(pdf): $(tex)
	pdflatex $(tex)
	make bib
	pdflatex $(tex)
	pdflatex $(tex)
	pdflatex $(tex)

bib: $(bib)
	$(bibtex) $(aux)

clean:
	$(RM) $(rmlist)

fold: $(tex)
	cp $(tex) $(tex).bkp
	fold -w 80 -s $(tex) > $(tex).tmp
	mv -f $(tex).tmp $(tex)
	make $(pdf)

touch:
	touch $(tex)
