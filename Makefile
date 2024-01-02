output: main.o my_math.o neural.o file.o
	gcc main.o my_math.o neural.o file.o -o output -lm

main.o: main.c main.h
	gcc -c main.c

my_math.o: my_math.c my_math.h
	gcc -c my_math.c

neural.o: neural.c neural.h
	gcc -c neural.c

file.o: file.c file.h
	gcc -c file.c

clean:
	rm *.o output
