XOR.java
--------
This example shows how to use Joone core engine writing Java code.
Read the Developer Guide to learn in details how does work its source code.
To runnit:
java -cp joone-engine.jar org.joone.samples.xor.XOR <inputFile> <errorFile>
where <inputFile> is the input file name (with its complete path) that contains the XOR truth table (xor.txt)
      <errorFile> is the output file name (with its complete path) that will contain the error values of the net

XORMemory.java
--------------
Sample class to demostrate the use of the MemoryInputSynapse class
to train the net with the patterns contained in a 2D array of doubles.
To runnit:
java -cp joone-engine.jar org.joone.samples.xor.XORMemory

EmbeddedXOR.java
----------------
This example shows the use of a neural network embedded in another
application that gets the output from the MemoryOutputSynapse class
giving to the net a set of predefined input patterns using the 
MemoryInputSynapse class.
To runnit:
java -cp joone-engine.jar org.joone.samples.xor.EmbeddedXOR.java xor.snet
where xor.snet is a serialized XOR neural network (i.e. obtained from the
GUI editor with File->Export menu item)

ImmediateEmbeddedXOR.java
-------------------------
This example shows the use of a neural network embedded in another
application that gets the output from the MemoryOutputSynapse class
giving to the net only one input patterns each time, using the 
DirectSynapse class.
To runnit:
java -cp joone-engine.jar org.joone.samples.xor.ImmediateEmbeddedXOR.java xor.snet
where xor.snet is a serialized XOR neural network (i.e. obtained from the
GUI editor with File->Export menu item)
