/*--- formatted by Jindent 2.1, (www.c-lab.de/~jindent) ---*/

/*
 * JOONE - Java Object Oriented Neural Engine
 * http://joone.sourceforge.net
 *
 */
package org.joone.samples.engine.multipleInputs;

import java.io.File;
import org.joone.engine.*;
import org.joone.engine.learning.*;
import org.joone.io.*;
import org.joone.net.*;

/**
 * Sample class to demostrate the use of the MultipleInputSynapse
 *
 *
 */
public class XOR_inputSwitch implements NeuralNetListener {
    private NeuralNet			nnet = null;
    private FileInputSynapse inputSynapse1, inputSynapse2, inputSynapse3;
    private FileInputSynapse desiredSynapse1, desiredSynapse2, desiredSynapse3;
    //    private MemoryOutputSynapse outputSynapse;
    private InputSwitchSynapse inputSw, desiredSw;
    private static String inputFile = "/tmp/xor.txt";
    
    /**
     * @param args the command line arguments
     */
    public static void main(String args[]) {
        XOR_inputSwitch xor = new XOR_inputSwitch();
        
        xor.initNeuralNet();
        xor.train();
    }
    
    /**
     * Method declaration
     */
    public void train() {
        
        // get the monitor object to train or feed forward
        Monitor monitor = nnet.getMonitor();
        
        // set the monitor parameters
        monitor.setLearningRate(0.8);
        monitor.setMomentum(0.3);
        monitor.setTrainingPatterns(4);
        monitor.setTotCicles(1000);
        monitor.setLearning(true);
        nnet.addNeuralNetListener(this);
//        nnet.start();
//        nnet.getMonitor().Go();
//        nnet.join();
//        System.out.println("Network stopped. Last RMSE="+nnet.getMonitor().getGlobalError());
        interrogate();
        interrogate();
        interrogate();
    }
    
    private void interrogate() {
        // neuralNet is an instance of NeuralNet
        Monitor monitor=nnet.getMonitor();
        monitor.setTotCicles(1);
        monitor.setLearning(false);
        FileOutputSynapse output=new FileOutputSynapse();
        // set the output synapse to write the output of the net
        output.setFileName("/tmp/xorOut.txt");
        // inject the input and get the output
        if(nnet!=null) {
            nnet.addOutputSynapse(output);
            System.out.println(nnet.check());
            nnet.start();
            monitor.Go();
            nnet.join();
        }
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
        
        input.setLayerName("inputLayer");
        hidden.setLayerName("hiddenLayer");
        output.setLayerName("outputLayer");
        
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
        inputSynapse1 = new FileInputSynapse();
        inputSynapse2 = new FileInputSynapse();
        inputSynapse3 = new FileInputSynapse();
        
        inputSynapse1.setInputFile(new File(inputFile));
        inputSynapse1.setName("input1");
        inputSynapse1.setAdvancedColumnSelector("1-2");
        inputSynapse1.setFirstRow(1);
        inputSynapse1.setLastRow(4);
        
        inputSynapse2.setInputFile(new File(inputFile));
        inputSynapse2.setName("input2");
        inputSynapse2.setAdvancedColumnSelector("1-2");
        inputSynapse2.setFirstRow(2);
        inputSynapse2.setLastRow(3);
        
        inputSynapse3.setInputFile(new File(inputFile));
        inputSynapse3.setName("input3");
        inputSynapse3.setAdvancedColumnSelector("1-2");
        inputSynapse3.setFirstRow(4);
        inputSynapse3.setLastRow(4);
        
        inputSw = new InputSwitchSynapse();
        inputSw.addInputSynapse(inputSynapse1);
        inputSw.addInputSynapse(inputSynapse2);
        inputSw.addInputSynapse(inputSynapse3);
        
        input.addInputSynapse(inputSw);
        
        // The Trainer and its desired output
        desiredSynapse1 = new FileInputSynapse();
        desiredSynapse2 = new FileInputSynapse();
        desiredSynapse3 = new FileInputSynapse();
        
        desiredSynapse1.setInputFile(new File(inputFile));
        desiredSynapse1.setName("desired1");
        desiredSynapse1.setAdvancedColumnSelector("3");
        desiredSynapse1.setFirstRow(1);
        desiredSynapse1.setLastRow(4);
        
        desiredSynapse2.setInputFile(new File(inputFile));
        desiredSynapse2.setName("desired2");
        desiredSynapse2.setAdvancedColumnSelector("3");
        desiredSynapse2.setFirstRow(2);
        desiredSynapse2.setLastRow(3);
        
        desiredSynapse3.setInputFile(new File(inputFile));
        desiredSynapse3.setName("desired3");
        desiredSynapse3.setAdvancedColumnSelector("3");
        desiredSynapse3.setFirstRow(4);
        desiredSynapse3.setLastRow(4);
        
        desiredSw = new InputSwitchSynapse();
        desiredSw.addInputSynapse(desiredSynapse1);
        desiredSw.addInputSynapse(desiredSynapse2);
        desiredSw.addInputSynapse(desiredSynapse3);
        
        TeachingSynapse trainer = new TeachingSynapse();
        trainer.setDesired(desiredSw);
        
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
        System.out.println("Cycle: "+(mon.getTotCicles()-mon.getCurrentCicle())+" RMSE:"+mon.getGlobalError());
    }
    
    public void netStarted(NeuralNetEvent e) {
        System.out.println("Training...");
    }
    
    public void netStopped(NeuralNetEvent e) {
        System.out.println("Stopped");
    }
    
    public void netStoppedError(NeuralNetEvent e, String error) {
    }
    
}



/*--- formatting done in "JMRA based on Sun Java Convention" style on 05-25-2002 ---*/

