/*
 * ValidationSample.java
 *
 * Created on 11 november 2002, 22.59
 * @author  pmarrone
 */

package org.joone.samples.engine.validation;

import org.joone.engine.*;
import org.joone.engine.learning.*;
import org.joone.net.*;
import org.joone.io.*;
import org.joone.util.*;

import java.io.*;

/**
 * This example shows how to check the training level of a neural network
 * using a validation data source.
 * The training and the validation phases of the created network is executed
 * many times, showing for each one the resulting RMSE.
 * This program shows how to build the same kind of neural net as that
 * contained into the org/joone/samples/editor/scripting/ValidationSample.ser
 * file using only java code and the core engine's API. Open that net in
 * the GUI editor to see the architecture of the net built in this example.
 */
public class MultipleValidationSample implements NeuralValidationListener {
    
    NeuralNet nnet;
    boolean ready;
    int totNets = 10; // Number of neural nets to train & validate
    int returnedNets = 0;
    double totRMSE = 0;
    double minRMSE = 99;
    
    long mStart;
    int trainingLCP = 1;
    int validationLCP = 16;
    int totCycles = 1000;
    FileWriter wr = null;
    
    private static String filePath = "org/joone/samples/engine/validation";
    // Must point to a trained XOR network without I/O components
    String xorNet = filePath+"/trainedXOR.snet";
    
    /** Creates a new instance of SampleScript */
    public MultipleValidationSample() {
    }
    
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        MultipleValidationSample sampleNet = new MultipleValidationSample();
        sampleNet.start();
    }
    
    private void start() {
        try{
            wr = new FileWriter(new File("/tmp/memory.txt"));
            while (trainingLCP <= validationLCP){
                // Start the LC Calculation
                startValidation(trainingLCP,validationLCP);
                trainingLCP += 1;
                wr.flush();
            }
            // Draws the error's curve
            wr.close();
        }
        catch (IOException ioe){ioe.printStackTrace();}
        System.out.println("Done.");
        System.exit(0);
        
    }
    
    private synchronized void startValidation(int trnP, int valP) {
        nnet = initializeModularParity(trnP, valP);
        //nnet = initializeSimpleParity(trnP, valP);
        //nnet = initializeNetworkI(trnP, valP);
        nnet.getMonitor().setTrainingPatterns(trnP);
        nnet.getMonitor().setValidationPatterns(valP);
        
        try {
            mStart = System.currentTimeMillis();
            returnedNets = 0;
            totRMSE = 0;
            minRMSE = 99;
            // n = total number of neural networks to create, train and validate
            int n = totNets;
            // First of all, starts the initial number of
            // neural networks that must be trained in parallel
            for (int i=0; i < 1; ++i)  {
                test(n--);
            }
            while (n > 0)   {
                // Waits for a neural network's validation termination
                // before to start another one
                while (!ready) {
                    try  {
                        wait();
                    } catch (InterruptedException doNothing) {}
                }
                ready = false;
                test(n--);
                long mem = getMemoryUse();
                wr.write(mem+"\r\n");
            }
            while (returnedNets < totNets){
                try  {
                    wait();
                } catch (InterruptedException doNothing) {}
            }
            // This code is executed when all the neural networks
            // have been trained and validated
            displayResults();
        } catch (IOException ioe) { ioe.printStackTrace(); }
    }
    
    // Run a new training & validation phase
    private void test(int n)  {
        nnet.randomize(0.5);
        nnet.setParam("ID", new Integer(n)); // Set its param ID
        // Create the trainer object
        NeuralNetTrainer trainer = new NeuralNetTrainer(nnet);
        //NeuralNetTester trainer = new NeuralNetTester(nnet,true,0);
        // Registers itself as a listener of the trainer object
        trainer.addValidationListener(this);
        // Run the training+validation tasks
        trainer.start();
    }
    
    /* This method is called by the trainers for each validated neural network
     * The param ID is used to recognize the returned net
     */
    public synchronized void netValidated(NeuralValidationEvent event) {
        // Shows the RMSE at the end of the validation phase
        NeuralNet NN = (NeuralNet)event.getSource();
        int n = ((Integer)NN.getParam("ID")).intValue();
        double rmse = NN.getMonitor().getGlobalError();
        //System.out.print("Returned NeuralNet #"+n);
        //System.out.println(" Validation RMSE: "+rmse);
        totRMSE += rmse;
        if (minRMSE > rmse)
            minRMSE = rmse;
        ++returnedNets;
        ready = true;
        notifyAll();
    }
    
    private void displayResults(){
        // This code is executed when all the neural networks have been trained and validated
        double aveRMSE = totRMSE/totNets;
        long mTot = System.currentTimeMillis()-mStart;
        System.out.println("---------------------------------------------------------");
        System.out.println("Training Patterns: "+trainingLCP);
        System.out.println("Average Generalization Error: "+aveRMSE);
        System.out.println("Minimum Generalization Error: "+minRMSE);
        System.out.println("Elapsed Time: "+mTot+" Miliseconds");
        System.out.println("---------------------------------------------------------");
    }
    
    /// Garbage collection ////
    private static long fSLEEP_INTERVAL = 20;
    
    private static long getMemoryUse(){
        //  collectGarbage(); // NOTE: To obtain the memory allocation w/o GC, comment this line.
        long totalMemory = Runtime.getRuntime().totalMemory();
        long freeMemory = Runtime.getRuntime().freeMemory();
        return (totalMemory - freeMemory);
    }
    
    
    private static void collectGarbage() {
        try {
            System.gc();
            Thread.currentThread().sleep(fSLEEP_INTERVAL);
            System.runFinalization();
            Thread.currentThread().sleep(fSLEEP_INTERVAL);
        }
        catch (InterruptedException ex){
            ex.printStackTrace();
        }
    }
    /** Configures & Starts the SimpleParity Network - Class Method
     * @param learningPatternNumber Number of Learning Patterns
     * @param testPatternNumber Number of Test Patterns
     */
    private NeuralNet initializeSimpleParity(int learningPatternNumber,int testPatternNumber){
        // Initialize the neural network
        NeuralNet network = new NeuralNet();
        // Define & Initialize the Learning & Test inputs
        double[][] learningData = constructLearningData(learningPatternNumber);
        double[][] testData = constructTestData(testPatternNumber);
        // Define & Initialize the network layers and Define the layer names
        LinearLayer input = new LinearLayer();
        SigmoidLayer hidden = new SigmoidLayer();
        SigmoidLayer output = new SigmoidLayer();
        input.setLayerName("Input Layer");
        hidden.setLayerName("Hidden Layer");
        output.setLayerName("Output Layer");
        // Define the number of neurons for each layer
        input.setRows(4);
        hidden.setRows(4);
        output.setRows(1);
        // Define the input -> hidden connection
        FullSynapse synapseIH = new FullSynapse();
        synapseIH.setName("IH Synapse");
        // Define the hidden -> output connection
        FullSynapse synapseHO = new FullSynapse();
        synapseHO.setName("HO Synapse");
        // Connect the Input Layer with the Hidden Layer
        NeuralNetFactory.connect(input,synapseIH,hidden);
        // Connect the Hidden Layer with the Output Layer
        NeuralNetFactory.connect(hidden,synapseHO,output);
        // Define & Initialize the Learning Input Synapse
        MemoryInputSynapse learningInputSynapse = NeuralNetFactory.createInput("Learning Input Synapse",learningData,1,1,4);
        // Define the Test Input Synapse
        MemoryInputSynapse testInputSynapse = NeuralNetFactory.createInput("Test Input Synapse",testData,1,1,4);
        // Initialize the Input Switch Synapse
        LearningSwitch inputSwitch = NeuralNetFactory.createSwitch("Input Switch Synapse",learningInputSynapse,testInputSynapse);
        // Connect the Input Switch Synapse to the Input Layer
        input.addInputSynapse(inputSwitch);
        // Define the Trainer Input Switch
        MemoryInputSynapse learningDesiredSynapse = NeuralNetFactory.createInput("Learning Desired Synapse",learningData,1,5,5);
        // Define the Test Input Synapse
        MemoryInputSynapse testDesiredSynapse = NeuralNetFactory.createInput("Test Desired Synapse",testData,1,5,5);
        // Initialize the Input Switch Synapse
        LearningSwitch learningSwitch = NeuralNetFactory.createSwitch("Learning Switch Synapse",learningDesiredSynapse,testDesiredSynapse);
        // Define the Trainer and link it to the Monitor
        TeachingSynapse trainer = new TeachingSynapse();
        trainer.setName("Simple Parity Trainer Synapse");
        // Connect the Teacher to the Output Layer
        output.addOutputSynapse(trainer);
        // Connect the Learning Switch Synapse to the Trainer
        trainer.setDesired(learningSwitch);
        // Define the Output Synapse Memory (Data)
        MemoryOutputSynapse outputMemoryData = new MemoryOutputSynapse();
        outputMemoryData.setName("Output Data");
        // Connect the Output Memory Synapse (Data) to the Output
        output.addOutputSynapse(outputMemoryData);
        // Incorpore the network components to the NeuralNet object
        network.addLayer(input);
        network.addLayer(hidden);
        network.addLayer(output);
        network.setTeacher(trainer);
        network.getMonitor().setLearningRate(0.7);
        network.getMonitor().setMomentum(0.5);
        network.getMonitor().setTotCicles(totCycles);
        return network;
    }
    /** Constructs the network Learning Data based on the GUI options - Class Method
     * @param learningPatternNumber The int Number of Training Patterns
     * @return The Training Patterns Vector
     */
    private double[][] constructLearningData(int learningPatternNumber){
        // Define the number of columns
        int columns = 5;
        // Define the Learning Data array
        double[][] learningData = new double[learningPatternNumber][columns];
        // Define the randomized Learning Data
        // Select the not Randomized patterns
        
        // The Simple Parity Input Data
        double[][] simpleParityData = {
            {0.0,0.0,0.0,0.0,1.0},
            {0.0,0.0,0.0,1.0,0.0},
            {0.0,0.0,1.0,0.0,0.0},
            {0.0,0.0,1.0,1.0,1.0},
            {0.0,1.0,0.0,0.0,0.0},
            {0.0,1.0,0.0,1.0,1.0},
            {0.0,1.0,1.0,0.0,1.0},
            {0.0,1.0,1.0,1.0,0.0},
            {1.0,0.0,0.0,0.0,0.0},
            {1.0,0.0,0.0,1.0,1.0},
            {1.0,0.0,1.0,0.0,1.0},
            {1.0,0.0,1.0,1.0,0.0},
            {1.0,1.0,0.0,0.0,1.0},
            {1.0,1.0,0.0,1.0,0.0},
            {1.0,1.0,1.0,0.0,0.0},
            {1.0,1.0,1.0,1.0,1.0}};
            
        for (int i = 0;i<learningPatternNumber;i++){
            // Construct the each pattern of the Learning Data
            learningData[i][0] = simpleParityData[i][0];
            learningData[i][1] = simpleParityData[i][1];
            learningData[i][2] = simpleParityData[i][2];
            learningData[i][3] = simpleParityData[i][3];
            learningData[i][4] = simpleParityData[i][4];
        }
        return learningData;
    }
    
    /** Constructs the network Test Data based on the GUI options - Class Method
     * @param testPatternNumber The int Number of Test Patterns
     * @return The Test Patterns Vector
     */
    private double[][] constructTestData(int testPatternNumber){
        // Define the number of columns
        int columns = 5;
        
        // Define the Learning Data array
        double[][] testData = new double[testPatternNumber][columns];
        // Define the randomized Learning Data
        // Select the not Randomized patterns
        // The Simple Parity Input Data
        double[][] simpleParityData = {{0.0,0.0,0.0,0.0,1.0},{0.0,0.0,0.0,1.0,0.0},{0.0,0.0,1.0,0.0,0.0},{0.0,0.0,1.0,1.0,1.0},{0.0,1.0,0.0,0.0,0.0},{0.0,1.0,0.0,1.0,1.0},{0.0,1.0,1.0,0.0,1.0},{0.0,1.0,1.0,1.0,0.0},{1.0,0.0,0.0,0.0,0.0},{1.0,0.0,0.0,1.0,1.0},{1.0,0.0,1.0,0.0,1.0},{1.0,0.0,1.0,1.0,0.0},{1.0,1.0,0.0,0.0,1.0},{1.0,1.0,0.0,1.0,0.0},{1.0,1.0,1.0,0.0,0.0},{1.0,1.0,1.0,1.0,1.0}};
        for (int i = 0;i<testPatternNumber;i++){
            // Construct the each pattern of the Test Data
            testData[i][0] = simpleParityData[i][0];
            testData[i][1] = simpleParityData[i][1];
            testData[i][2] = simpleParityData[i][2];
            testData[i][3] = simpleParityData[i][3];
            testData[i][4] = simpleParityData[i][4];
        }
        return testData;
    }
    
    private NeuralNet initializeModularParity(int learningPatternNumber,int testPatternNumber){
        // Initialize the neural networks
        NeuralNet network = new NeuralNet();
        NestedNeuralLayer firstNetwork = new NestedNeuralLayer();
        NestedNeuralLayer secondNetwork = new NestedNeuralLayer();
        // Set the First & Second network properties
        firstNetwork.setNeuralNet(xorNet);
        secondNetwork.setNeuralNet(xorNet);
        firstNetwork.setLayerName("First Network");
        secondNetwork.setLayerName("Second Network");
        //firstNetwork.setLearning(true);
        //secondNetwork.setLearning(true);
        // Define & Initialize the Learning & Test inputs
        double[][] learningData=constructLearningData(learningPatternNumber);
        double[][] testData=constructTestData(testPatternNumber);
        // Define & Initialize the network layers and Define the layer names
        LinearLayer inputFirst=new LinearLayer();
        LinearLayer inputSecond=new LinearLayer();
        SigmoidLayer hidden=new SigmoidLayer();
        SigmoidLayer output=new SigmoidLayer();
        inputFirst.setLayerName("First Input Third Network Layer");
        inputSecond.setLayerName("Second Input Third Network Layer");
        hidden.setLayerName("Hidden Third Network Layer");
        output.setLayerName("Output Third Network Layer");
        // Define the number of neurons for each layer
        inputFirst.setRows(1);
        inputSecond.setRows(1);
        hidden.setRows(2);
        output.setRows(1);
        // Define the first network output -> input connection for the third network
        DirectSynapse firstSynapseOI=new DirectSynapse();
        firstSynapseOI.setName("First OI Synapse");
        // Define the second network output -> input connection for the third network
        DirectSynapse secondSynapseOI=new DirectSynapse();
        secondSynapseOI.setName("First OI Synapse");
        // Define the first input -> hidden connection for the third network
        FullSynapse firstSynapseIH=new FullSynapse();
        firstSynapseIH.setName("First IH Synapse");
        // Define the Second input -> hidden connection for the third network
        FullSynapse secondSynapseIH=new FullSynapse();
        secondSynapseIH.setName("Second IH Synapse");
        // Define the hidden -> output connection for the third network
        FullSynapse synapseHO=new FullSynapse();
        synapseHO.setName("HO Synapse");
        // Connect the First Network with the First Input Third Network Layer
        NeuralNetFactory.connect(firstNetwork,firstSynapseOI,inputFirst);
        // Connect the Second Network with the Second Input Third Network Layer
        NeuralNetFactory.connect(secondNetwork,secondSynapseOI,inputSecond);
        // Connect the First Input Third Network Layer with the Hidden Third Network Layer
        NeuralNetFactory.connect(inputFirst,firstSynapseIH,hidden);
        // Connect the Second Input Third Network Layer with the Hidden Third Network Layer
        NeuralNetFactory.connect(inputSecond,secondSynapseIH,hidden);
        // Connect the Hidden Third Network Layer with the Output Third Network Layer
        NeuralNetFactory.connect(hidden,synapseHO,output);
        // Define & Initialize the First Learning Input Synapse
        MemoryInputSynapse firstLearningInputSynapse=NeuralNetFactory.createInput("First Learning Input Synapse",learningData,1,1,2);
        // Define the First Test Input Synapse
        MemoryInputSynapse firstTestInputSynapse=NeuralNetFactory.createInput("First Test Input Synapse",testData,1,1,2);
        // Initialize the First Input Switch Synapse
        LearningSwitch firstInputSwitch=NeuralNetFactory.createSwitch("First Input Switch Synapse",firstLearningInputSynapse,firstTestInputSynapse);
        // Connect the First Input Switch Synapse to the First Network
        firstNetwork.addInputSynapse(firstInputSwitch);
        // Define & Initialize the Second Learning Input Synapse
        MemoryInputSynapse secondLearningInputSynapse=NeuralNetFactory.createInput("Second Learning Input Synapse",learningData,1,3,4);
        // Define the Second Test Input Synapse
        MemoryInputSynapse secondTestInputSynapse=NeuralNetFactory.createInput("Second Test Input Synapse",testData,1,3,4);
        // Initialize the Second Input Switch Synapse
        LearningSwitch secondInputSwitch=NeuralNetFactory.createSwitch("Second Input Switch Synapse",secondLearningInputSynapse,secondTestInputSynapse);
        secondInputSwitch.setStepCounter(false);
        // Connect the Second Input Switch Synapse to the Second Network
        secondNetwork.addInputSynapse(secondInputSwitch);
        // Define the Trainer Input Switch
        MemoryInputSynapse learningDesiredSynapse=NeuralNetFactory.createInput("Learning Desired Synapse",learningData,1,5,5);
        // Define the Test Input Synapse
        MemoryInputSynapse testDesiredSynapse=NeuralNetFactory.createInput("Test Desired Synapse",testData,1,5,5);
        // Initialize the Input Switch Synapse
        LearningSwitch learningSwitch=NeuralNetFactory.createSwitch("Learning Switch Synapse",learningDesiredSynapse,testDesiredSynapse);
        // Define the Trainer and link it to the Monitor
        TeachingSynapse trainer=new TeachingSynapse();
        trainer.setName("Modular Parity Trainer Synapse");
        // Connect the Teacher to the Output Layer
        output.addOutputSynapse(trainer);
        // Connect the Learning Switch Synapse to the Trainer
        trainer.setDesired(learningSwitch);
        // Define the Output Synapse Memory (Data)
        MemoryOutputSynapse outputMemoryData=new MemoryOutputSynapse();
        outputMemoryData.setName("Output Data");
        // Connect the Output Memory Synapse (Data) to the Output Third Network Layer
        output.addOutputSynapse(outputMemoryData);
        // Incorpore the network components to the NeuralNet object
        network.addLayer(firstNetwork);
        network.addLayer(secondNetwork);
        network.addLayer(inputFirst);
        network.addLayer(inputSecond);
        network.addLayer(hidden);
        network.addLayer(output);
        network.setTeacher(trainer);
        network.getMonitor().setLearningRate(0.5);
        network.getMonitor().setMomentum(0.5);
        network.getMonitor().setTotCicles(totCycles);
        return network;
    }
    /** Initializes the EETNN Neural Network I - Class Method
     * @param learningPatternNumber Number of Learning Patterns
     * @param testPatternNumber Number of Test Patterns
     */
    private NeuralNet initializeNetworkI(int learningPatternNumber,int testPatternNumber){
        // Initialize the neural network
        NeuralNet network=new NeuralNet();
        // Define & Initialize the network layers and Define the layer names
        LinearLayer input=new LinearLayer();
        SigmoidLayer hidden=new SigmoidLayer();
        SigmoidLayer output=new SigmoidLayer();
        input.setLayerName("Input Layer");
        hidden.setLayerName("Hidden Layer");
        output.setLayerName("Output Layer");
        // Define the number of neurons for each layer
        input.setRows(2);
        hidden.setRows(2);
        output.setRows(1);
        // Define the input -> hidden connection
        FullSynapse synapseIH=new FullSynapse();
        synapseIH.setName("IH Synapse");
        // Define the hidden -> output connection
        FullSynapse synapseHO=new FullSynapse();
        synapseHO.setName("HO Synapse");
        // Connect the Input Layer with the Hidden Layer
        NeuralNetFactory.connect(input,synapseIH,hidden);
        // Connect the Hidden Layer with the Output Layer
        NeuralNetFactory.connect(hidden,synapseHO,output);
        // Define & Initialize the Learning Input Synapse
        XLSInputSynapse learningInputSynapse = new XLSInputSynapse();
        learningInputSynapse.setName("Learning Input Synapse");
        learningInputSynapse.setInputFile(new File("/tmp/wine.xls"));
        learningInputSynapse.setAdvancedColumnSelector("6,7");
        learningInputSynapse.setSheetName("wine.data");
        learningInputSynapse.setFirstRow(2);
        learningInputSynapse.setLastRow(100);
        // Define the Test Input Synapse
        XLSInputSynapse testInputSynapse = new XLSInputSynapse();
        testInputSynapse.setName("Test Input Synapse");
        testInputSynapse.setInputFile(new File("/tmp/wine.xls"));
        testInputSynapse.setAdvancedColumnSelector("6,7");
        testInputSynapse.setSheetName("wine.data");
        testInputSynapse.setFirstRow(2);
        testInputSynapse.setLastRow(100);
        // Initialize the Input Switch Synapse
        LearningSwitch inputSwitch=NeuralNetFactory.createSwitch("Input Switch Synapse",learningInputSynapse,testInputSynapse);
        // Connect the Input Switch Synapse to the Input Layer
        input.addInputSynapse(inputSwitch);
        // Define the Trainer's Learning Input Synapse
        XLSInputSynapse learningDesiredSynapse = new XLSInputSynapse();
        learningDesiredSynapse.setName("Learning Desired Synapse");
        learningDesiredSynapse.setInputFile(new File("/tmp/wine.xls"));
        learningDesiredSynapse.setAdvancedColumnSelector("8");
        learningDesiredSynapse.setSheetName("wine.data");
        learningDesiredSynapse.setFirstRow(2);
        learningDesiredSynapse.setLastRow(100);
        // Define the Trainer's Test Input Synapse
        XLSInputSynapse testDesiredSynapse = new XLSInputSynapse();
        testDesiredSynapse.setName("Test Desired Synapse");
        testDesiredSynapse.setInputFile(new File("/tmp/wine.xls"));
        testDesiredSynapse.setAdvancedColumnSelector("8");
        testDesiredSynapse.setSheetName("wine.data");
        testDesiredSynapse.setFirstRow(2);
        testDesiredSynapse.setLastRow(100);
        // Initialize the Input Switch Synapse
        LearningSwitch learningSwitch=NeuralNetFactory.createSwitch("Learning Switch Synapse",learningDesiredSynapse,testDesiredSynapse);
        // Define the Trainer and link it to the Monitor
        TeachingSynapse trainer=new TeachingSynapse();
        trainer.setName("EETNN Trainer Synapse");
        // Connect the Teacher to the Output Layer
        output.addOutputSynapse(trainer);
        // Connect the Learning Switch Synapse to the Trainer
        trainer.setDesired(learningSwitch);
        // Define the Output Synapse Memory (Data)
    /*XLSOutputSynapse outputMemoryData=new XLSOutputSynapse();
    outputMemoryData.setName("Output Data");
    outputMemoryData.setFileName("/tmp/OutputData.xls");
    outputMemoryData.setSheetName("wine.data");
    // Connect the Output Memory Synapse (Data) to the Output
    output.addOutputSynapse(outputMemoryData); */
        // Incorpore the network components to the NeuralNet object
        network.addLayer(input);
        network.addLayer(hidden);
        network.addLayer(output);
        network.setTeacher(trainer);
        return network;
    }
    
}
