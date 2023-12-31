output: main.o math.o neural.o file.o
	gcc main.o math.o neural.o file.o -o output

main.o: main.c main.h
	gcc -c main.c

math.o: math.c math.h
	gcc -c math.c

neural.o: neural.c neural.h
	gcc -c neural.c

file.o: file.c file.h
	gcc -c file.c

clean:
	rm *.o output
