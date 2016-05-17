# Target rules
all: build

build: mc_s mc_o

mc_s:
	gcc -msse2 -DDSFMT_MEXP=521 -DHAVE_SSE2 -o mc_s dsfmt/dSFMT.c single.c

mc_o:
	gcc -msse2 -fopenmp -DDSFMT_MEXP=521 -DHAVE_SSE2 -o mc_o dsfmt/dSFMT.c omp.c

clean:
	rm -f mc_s mc_o

clobber: clean
