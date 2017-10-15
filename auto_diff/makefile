all: test
test: Graph.o Node.o Add.o Input.o Mult.o test.o
	g++ Graph.o Node.o Add.o Input.o Mult.o test.o -o test
test.o: test.cpp Graph.o Node.o Add.o Input.o Mult.o
	g++ -c test.cpp
Graph.o: Graph.cpp Node.o
	g++ -c Graph.cpp
Mult.o: Mult.cpp Node.o
	g++ -c Mult.cpp
Add.o: Add.cpp Node.o
	g++ -c Add.cpp
Input.o: Input.cpp Node.o
	g++ -c Input.cpp
Node.o: Node.cpp
	g++ -c Node.cpp
clean:
	rm *o test
