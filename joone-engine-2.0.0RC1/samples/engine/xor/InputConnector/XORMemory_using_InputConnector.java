/*--- formatted by Jindent 2.1, (www.c-lab.de/~jindent) ---*/

/*
 * JOONE - Java Object Oriented Neural Engine
 * http://joone.sourceforge.net
 *
 * XORMemory.java
 *
 */
package org.joone.samples.engine.xor.InputConnector;

import org.joone.engine.*;
import org.joone.engine.learning.*;
import org.joone.io.*;
import org.joone.net.NeuralNet;
import org.joone.util.*;

/**
 * Sample class to demostrate the use of the MemoryInputSynapse
 *
 * @author P.Marrone
 */
public class XORMemory_using_InputConnector implements NeuralNetListener {
    
    // XOR input
    private double[][]  inputArray = new double[][] {
        {0.0, 0.0, 0.0},
        {0.0, 1.0, 1.0},
        {1.0, 0.0, 1.0},
        {1.0, 1.0, 0.0}
    };
    
    private long mills;
    /**
     * @param args the command line arguments
     */
    public static void main(String args[]) {
        XORMemory_using_InputConnector   xor = new XORMemory_using_InputConnector();
        
        xor.Go();
    }
    
    /**
     * Method declaration
     */
    public void Go() {
        // Firts, creates the three Layers
        LinearLayer	input = new LinearLayer();
        SigmoidLayer	hidden = new SigmoidLayer();
        SigmoidLayer	output = new SigmoidLayer();
        
        input.setLayerName("input");
        hidden.setLayerName("hidden");
        output.setLayerName("output");
        
        // sets their dimensions
        input.setRows(2);
        hidden.setRows(3);
        output.setRows(1);
        
        // Now create the two Synapses
        FullSynapse synapse_IH = new FullSynapse();	/* input -> hidden conn. */
        FullSynapse synapse_HO = new FullSynapse();	/* hidden -> output conn. */
        
        synapse_IH.setName("IH");
        synapse_HO.setName("HO");
        
        // Connect the input layer whit the hidden layer
        input.addOutputSynapse(synapse_IH);
        hidden.addInputSynapse(synapse_IH);
        
        // Connect the hidden layer whit the output layer
        hidden.addOutputSynapse(synapse_HO);
        output.addInputSynapse(synapse_HO);
                                
        // Set the input data 
        // as we want to use the InputConnectors, we read all the three input columns
        MemoryInputSynapse  inputStream = new MemoryInputSynapse();
        inputStream.setInputArray(inputArray);
        inputStream.setAdvancedColumnSelector("1-3");
        //inputStream.addPlugIn(new ShufflePlugin());
        
        /* The first InputConnector containing the training data */
        InputConnector input1 = new InputConnector();
        input1.setInputSynapse(inputStream);
        // The first two columns contain the input values
        input1.setAdvancedColumnSelector("1,2");
        //input1.setBuffered(true); // By default it's false
        input.addInputSynapse(input1);

        /* The second InputConnector containing the desired data */
        InputConnector input2 = new InputConnector();
        input2.setInputSynapse(inputStream);
        // The last column contains the desired values
        input2.setAdvancedColumnSelector("3");
        //input2.setBuffered(true); // By default it's false

        TeachingSynapse trainer = new TeachingSynapse();
        trainer.setDesired(input2);
        
        // Connects the Teacher to the last layer of the net
        output.addOutputSynapse(trainer);
        
        NeuralNet nnet = new NeuralNet();
        nnet.addLayer(input, NeuralNet.INPUT_LAYER);
        nnet.addLayer(hidden, NeuralNet.HIDDEN_LAYER);
        nnet.addLayer(output, NeuralNet.OUTPUT_LAYER);
        nnet.setTeacher(trainer);
        // Gets the Monitor object and set the learning parameters
        Monitor monitor = nnet.getMonitor();
        monitor.setLearningRate(0.8);
        monitor.setMomentum(0.3);
        
        monitor.setTrainingPatterns(4);	// # of rows (patterns) contained in the input file
        monitor.setTotCicles(2000);		// How many times the net must be trained on the input patterns
        monitor.setLearning(true);		// The net must be trained
        // The application registers itself as monitor's listener so it can receive
        // the notifications of termination from the net.
        monitor.addNeuralNetListener(this);
        mills = System.currentTimeMillis();
        nnet.go();					// The net starts the training job
    }
    
    /**
     * Method declaration
     */
    public void netStopped(NeuralNetEvent e) {
        long delay = System.currentTimeMillis() - mills;
        System.out.println("Training finished after "+delay+" ms");
        System.exit(0);
    }
    
    /**
     * Method declaration
     */
    public void cicleTerminated(NeuralNetEvent e) {
    }
    
    /**
     * Method declaration
     */
    public void netStarted(NeuralNetEvent e) {
        System.out.println("Training...");
    }
    
    public void errorChanged(NeuralNetEvent e) {
        Monitor mon = (Monitor) e.getSource();
        long	c = mon.getCurrentCicle();
        
        // We want to print the results every 200 epochs
        if ((c % 200) == 0) {
            System.out.println(c + " epochs remaining - RMSE = " + mon.getGlobalError());
        }
    }
    
    public void netStoppedError(NeuralNetEvent e,String error) {
    }
    
}



/*--- formatting done in "JMRA based on Sun Java Convention" style on 05-08-2002 ---*/

