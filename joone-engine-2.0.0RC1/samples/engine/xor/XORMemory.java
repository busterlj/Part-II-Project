/*--- formatted by Jindent 2.1, (www.c-lab.de/~jindent) ---*/

/*
 * JOONE - Java Object Oriented Neural Engine
 * http://joone.sourceforge.net
 *
 * XORMemory.java
 *
 */
package org.joone.samples.engine.xor;

import org.joone.engine.*;
import org.joone.engine.learning.*;
import org.joone.io.*;
import org.joone.net.NeuralNet;

/**
 * Sample class to demostrate the use of the MemoryInputSynapse
 *
 * @author Jos�?Rodriguez
 */
public class XORMemory implements NeuralNetListener {
    
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
        XORMemory   xor = new XORMemory();
        if (args.length == 0)
           xor.Go(10000);
        else
           xor.Go(new Integer(args[0]).intValue());
    }
    
    /**
     * Method declaration
     */
    public void Go(int epochs) {
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
                        
        MemoryInputSynapse  inputStream = new MemoryInputSynapse();
        
        // The first two columns contain the input values
        inputStream.setInputArray(inputArray);
        inputStream.setAdvancedColumnSelector("1,2");
        
        // set the input data
        input.addInputSynapse(inputStream);
        
        TeachingSynapse trainer = new TeachingSynapse();
        
        // Setting of the file containing the desired responses provided by a FileInputSynapse
        MemoryInputSynapse samples = new MemoryInputSynapse();
        
        
        // The output values are on the third column of the file
        samples.setInputArray(inputArray);
        samples.setAdvancedColumnSelector("3");
        trainer.setDesired(samples);
        
        // Connects the Teacher to the last layer of the net
        output.addOutputSynapse(trainer);

        // Creates a new NeuralNet
        NeuralNet nnet = new NeuralNet();
        /*
         * All the layers must be inserted in the NeuralNet object
         */
        nnet.addLayer(input, NeuralNet.INPUT_LAYER);
        nnet.addLayer(hidden, NeuralNet.HIDDEN_LAYER);
        nnet.addLayer(output, NeuralNet.OUTPUT_LAYER);
        Monitor monitor = nnet.getMonitor();
        monitor.setTrainingPatterns(4);	// # of rows (patterns) contained in the input file
        monitor.setTotCicles(epochs);	// How many times the net must be trained on the input patterns        
        monitor.setLearningRate(0.8);
        monitor.setMomentum(0.5);
        monitor.setLearning(true);	// The net must be trained
        monitor.setSingleThreadMode(true);  // Set to false for multi-thread mode
        /* The application registers itself as monitor's listener so it can receive
          the notifications of termination from the net. */
        monitor.addNeuralNetListener(this);
        mills = System.currentTimeMillis();
        nnet.go(); // The net starts in async mode
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
        long	cl = c / 1000;
        
        // We want to print the results every 1000 cycles
        if ((cl * 1000) == c) {
            System.out.println(c + " cycles remaining - Error = " + mon.getGlobalError());
        }
    }
    
    public void netStoppedError(NeuralNetEvent e,String error) {
    }
    
}



/*--- formatting done in "JMRA based on Sun Java Convention" style on 05-08-2002 ---*/

