all: test
test: Graph.o Node.o Input.o Mult.o Add.o Sigmoid.o Trace.o Tensor.o test.o
	g++ Graph.o Node.o Input.o Mult.o Add.o Sigmoid.o Trace.o Tensor.o test.o -o test
test.o: test.cpp Graph.o Node.o Input.o Mult.o Add.o Sigmoid.o Trace.o Tensor.o
	g++ -c test.cpp
Graph.o: Graph.cpp Node.o
	g++ -c Graph.cpp
Node.o: Node.cpp
	g++ -c Node.cpp
Mult.o: Mult.cpp Node.o
	g++ -c Mult.cpp
Add.o: Add.cpp Node.o
	g++ -c Add.cpp
Input.o: Input.cpp Node.o
	g++ -c Input.cpp
Sigmoid.o: Sigmoid.cpp Node.o
	g++ -c Sigmoid.cpp
Trace.o: Trace.cpp Node.o
	g++ -c Trace.cpp
Tensor.o: Tensor.cpp Node.o
	g++ -c Tensor.cpp
clean:
	rm *o test
