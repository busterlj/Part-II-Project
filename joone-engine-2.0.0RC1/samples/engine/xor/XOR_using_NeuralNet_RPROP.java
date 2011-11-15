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
import org.joone.net.*;
import java.util.Vector;

/**
 * Sample class to demostrate the use of the MemoryInputSynapse
 *
 * @author Jos�?Rodriguez
 */
public class XOR_using_NeuralNet_RPROP implements NeuralNetListener {
    private NeuralNet			nnet = null;
    private MemoryInputSynapse  inputSynapse, desiredOutputSynapse;
    private MemoryOutputSynapse outputSynapse;
    
    // XOR input
    private double[][]			inputArray = new double[][] {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };
    
    // XOR desired output
    private double[][]			desiredOutputArray = new double[][] {
        {0.0},
        {1.0},
        {1.0},
        {0.0}
    };
    
    /**
     * @param args the command line arguments
     */
    public static void main(String args[]) {
        XOR_using_NeuralNet_RPROP xor = new XOR_using_NeuralNet_RPROP();
        
        xor.initNeuralNet();
        xor.train();
    }
    
    /**
     * Method declaration
     */
    public void train() {
        
        // set the inputs
        inputSynapse.setInputArray(inputArray);
        inputSynapse.setAdvancedColumnSelector("1,2");
        
        // set the desired outputs
        desiredOutputSynapse.setInputArray(desiredOutputArray);
        desiredOutputSynapse.setAdvancedColumnSelector("1");
        
        // get the monitor object to train or feed forward
        Monitor monitor = nnet.getMonitor();
        
        // set the monitor parameters
        monitor.setLearningRate(1); // RPROP needs a positive LR, because
        // ExtendableLearner multiplies the gradient with the LR before passing
        // it to the RpropExtender. RPROP only looks at the sign of the gradient
        // so as long as the lR is positive, RPROP works correctly.
        
        monitor.setTrainingPatterns(inputArray.length);
        monitor.setTotCicles(500);
        // RPROP parameters
        monitor.addLearner(0, "org.joone.engine.RpropLearner");
        monitor.setBatchSize(monitor.getTrainingPatterns());
        monitor.setLearningMode(0);
        
        monitor.setLearning(true);
        nnet.addNeuralNetListener(this);
        nnet.go();
    }
    
    /**
     * Method declaration
     */
    protected void initNeuralNet() {
        
        // First create the three layers
        LinearLayer	input = new LinearLayer();
        SigmoidLayer	hidden = new SigmoidLayer();
        SigmoidLayer	output = new SigmoidLayer();
        
        // set the dimensions of the layers
        input.setRows(2);
        hidden.setRows(3);
        output.setRows(1);
        
        // Now create the two Synapses
        FullSynapse synapse_IH = new FullSynapse();	/* input -> hidden conn. */
        FullSynapse synapse_HO = new FullSynapse();	/* hidden -> output conn. */
        
        // Connect the input layer whit the hidden layer
        input.addOutputSynapse(synapse_IH);
        hidden.addInputSynapse(synapse_IH);
        
        // Connect the hidden layer whit the output layer
        hidden.addOutputSynapse(synapse_HO);
        output.addInputSynapse(synapse_HO);
        
        // the input to the neural net
        inputSynapse = new MemoryInputSynapse();
        
        input.addInputSynapse(inputSynapse);
        
        // the output of the neural net
        outputSynapse = new MemoryOutputSynapse();
        
        output.addOutputSynapse(outputSynapse);
        
        // The Trainer and its desired output
        desiredOutputSynapse = new MemoryInputSynapse();
        
        TeachingSynapse trainer = new TeachingSynapse();
        
        trainer.setDesired(desiredOutputSynapse);
        
        // Now we add this structure to a NeuralNet object
        nnet = new NeuralNet();
        
        nnet.addLayer(input, NeuralNet.INPUT_LAYER);
        nnet.addLayer(hidden, NeuralNet.HIDDEN_LAYER);
        nnet.addLayer(output, NeuralNet.OUTPUT_LAYER);
        nnet.setTeacher(trainer);
        output.addOutputSynapse(trainer);
    }
    
    public void cicleTerminated(NeuralNetEvent e) {
    }
    
    public void errorChanged(NeuralNetEvent e) {
        Monitor mon = (Monitor)e.getSource();
        if (mon.getCurrentCicle() % 100 == 0)
            System.out.println("Epoch: "+(mon.getTotCicles()-mon.getCurrentCicle())+" RMSE:"+mon.getGlobalError());
    }
    
    public void netStarted(NeuralNetEvent e) {
    }
    
    public void netStopped(NeuralNetEvent e) {
        Monitor mon = (Monitor)e.getSource();
        // Read the last pattern and print it out
        Vector patts = outputSynapse.getAllPatterns();
        Pattern pattern = (Pattern)patts.elementAt(patts.size() - 1);
        System.out.println("Output Pattern = " + pattern.getArray()[0] + " Error: " + mon.getGlobalError());
    }
    
    public void netStoppedError(NeuralNetEvent e, String error) {
    }
    
}



/*--- formatting done in "JMRA based on Sun Java Convention" style on 05-25-2002 ---*/

