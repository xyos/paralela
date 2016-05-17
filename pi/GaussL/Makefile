# Target rules
all: build

build: gaus_s gaus_o

gaus_s:
	gcc single.cc -lm -o gaus_s 

gaus_o:
	gcc omp.cc -lm -fopenmp -o gaus_o 

clean:
	rm -f gaus_s gaus_o

clobber: clean
