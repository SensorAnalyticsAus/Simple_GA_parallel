all: sga_parr

sga_parr: sga_parr.o obj3.o
	mpicc sga_parr.o obj3.o -o sga_parr -lm

sga_parr.o: sga_parr.c
	mpicc -c sga_parr.c

obj3.o: obj3.c
	gcc -c obj3.c -o obj3.o


clean:
	rm -f obj3.o sga_parr.o core *~
