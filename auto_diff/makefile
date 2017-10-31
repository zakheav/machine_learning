all: test
test: obj/test.o obj/Calculator.o obj/Graph.o obj/Node.o obj/Input.o obj/Mult.o obj/ScalarMult.o obj/SquareSum.o obj/Add.o obj/Sigmoid.o obj/Trace.o obj/Tensor.o
	g++ obj/test.o obj/Calculator.o obj/Graph.o obj/Node.o obj/Input.o obj/Mult.o obj/ScalarMult.o obj/SquareSum.o obj/Add.o obj/Sigmoid.o obj/Trace.o obj/Tensor.o -o test
obj/test.o: test.cpp obj/Calculator.o
	g++ -c test.cpp -o obj/test.o
obj/Graph.o: Graph.cpp obj/Node.o
	g++ -c Graph.cpp -o obj/Graph.o
obj/Calculator.o: Calculator.cpp obj/Graph.o obj/Node.o obj/Input.o obj/Mult.o obj/ScalarMult.o obj/SquareSum.o obj/Add.o obj/Sigmoid.o obj/Trace.o obj/Tensor.o
	g++ -c Calculator.cpp -o obj/Calculator.o
obj/Node.o: node/Node.cpp
	g++ -c node/Node.cpp -o obj/Node.o
obj/Mult.o: node/Mult.cpp obj/Node.o
	g++ -c node/Mult.cpp -o obj/Mult.o
obj/ScalarMult.o: node/ScalarMult.cpp obj/Node.o
	g++ -c node/ScalarMult.cpp -o obj/ScalarMult.o
obj/SquareSum.o: node/SquareSum.cpp obj/Node.o
	g++ -c node/SquareSum.cpp -o obj/SquareSum.o
obj/Add.o: node/Add.cpp obj/Node.o
	g++ -c node/Add.cpp -o obj/Add.o
obj/Input.o: node/Input.cpp obj/Node.o
	g++ -c node/Input.cpp -o obj/Input.o
obj/Sigmoid.o: node/Sigmoid.cpp obj/Node.o
	g++ -c node/Sigmoid.cpp -o obj/Sigmoid.o
obj/Trace.o: node/Trace.cpp obj/Node.o
	g++ -c node/Trace.cpp -o obj/Trace.o
obj/Tensor.o: Tensor.cpp obj/Node.o
	g++ -c Tensor.cpp -o obj/Tensor.o
clean:
	rm obj/*o test
