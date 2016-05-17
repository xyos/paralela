# Target rules
all: build

build: leib_s leib_o

leib_s:
	gcc single.cc -o leib_s

leib_o:
	gcc omp.cc -fopenmp -o leib_o

clean:
	rm -f leib_s leib_o

clobber: clean
