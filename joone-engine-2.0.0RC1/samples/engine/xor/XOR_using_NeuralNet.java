/*
 * JOONE - Java Object Oriented Neural Engine
 * http://joone.sourceforge.net
 *
 * XOR_using_NeuralNet.java
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
public class XOR_using_NeuralNet implements NeuralNetListener {
    private NeuralNet			nnet = null;
    private MemoryInputSynapse  inputSynapse, desiredOutputSynapse;
    private MemoryOutputSynapse outputSynapse;
    LinearLayer	input;
    SigmoidLayer hidden, output;
    boolean singleThreadMode = true;
    
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
        XOR_using_NeuralNet xor = new XOR_using_NeuralNet();
        
        xor.initNeuralNet();
        xor.train();
        xor.interrogate();
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
        monitor.setLearningRate(0.8);
        monitor.setMomentum(0.3);
        monitor.setTrainingPatterns(inputArray.length);
        monitor.setTotCicles(5000);
        monitor.setLearning(true);
        
        long initms = System.currentTimeMillis();
        // Run the network in single-thread, synchronized mode
        nnet.getMonitor().setSingleThreadMode(singleThreadMode);
        nnet.go(true);
        System.out.println("Total time= "+(System.currentTimeMillis() - initms)+" ms");
    }
    
    private void interrogate() {
        // set the inputs
        inputSynapse.setInputArray(inputArray);
        inputSynapse.setAdvancedColumnSelector("1,2");
        Monitor monitor=nnet.getMonitor();
        monitor.setTrainingPatterns(4);
        monitor.setTotCicles(1);
        monitor.setLearning(false);
        FileOutputSynapse foutput=new FileOutputSynapse();
        // set the output synapse to write the output of the net
        foutput.setFileName("/tmp/xorOut.txt");
        if(nnet!=null) {
            nnet.addOutputSynapse(foutput);
            System.out.println(nnet.check());
            nnet.getMonitor().setSingleThreadMode(singleThreadMode);
            nnet.go();
        }
    }
    
    /**
     * Method declaration
     */
    protected void initNeuralNet() {
        
        // First create the three layers
        input = new LinearLayer();
        hidden = new SigmoidLayer();
        output = new SigmoidLayer();
        
        // set the dimensions of the layers
        input.setRows(2);
        hidden.setRows(3);
        output.setRows(1);
        
        input.setLayerName("L.input");
        hidden.setLayerName("L.hidden");
        output.setLayerName("L.output");
        
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
        nnet.addNeuralNetListener(this);
    }
    
    public void cicleTerminated(NeuralNetEvent e) {
    }
    
    public void errorChanged(NeuralNetEvent e) {
        Monitor mon = (Monitor)e.getSource();
        if (mon.getCurrentCicle() % 100 == 0)
            System.out.println("Epoch: "+(mon.getTotCicles()-mon.getCurrentCicle())+" RMSE:"+mon.getGlobalError());
    }
    
    public void netStarted(NeuralNetEvent e) {
        Monitor mon = (Monitor)e.getSource();
        System.out.print("Network started for ");
        if (mon.isLearning())
            System.out.println("training.");
        else
            System.out.println("interrogation.");
    }
    
    public void netStopped(NeuralNetEvent e) {
        Monitor mon = (Monitor)e.getSource();
        System.out.println("Network stopped. Last RMSE="+mon.getGlobalError());
    }
    
    public void netStoppedError(NeuralNetEvent e, String error) {
        System.out.println("Network stopped due the following error: "+error);
    }
    
}
