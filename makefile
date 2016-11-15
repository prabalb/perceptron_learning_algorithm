OBJS = main.o pla.o
CC = g++
DEBUG = -g -O3
CFLAGS = -std=c++11 -Wall -fPIC -c $(DEBUG)
LFLAGS = -Wall $(DEBUG)

runProg : $(OBJS)
	$(CC) $(LFLAGS) $(OBJS) -o runProg

main.o : main.cpp pla.cpp pla.h
	$(CC) $(CFLAGS) main.cpp

pla.o : pla.cpp pla.h
	$(CC) $(CFLAGS) pla.cpp

clean:	
	\rm *.o runProg
