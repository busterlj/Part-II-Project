/*--- formatted by Jindent 2.1, (www.c-lab.de/~jindent) ---*/

/*
 * JOONE - Java Object Oriented Neural Engine
 * http://joone.sourceforge.net
 */
package org.joone.samples.engine.xor;

import java.io.Serializable;
import java.util.Vector;
import org.joone.engine.*;
import org.joone.engine.learning.*;
import org.joone.engine.listeners.*;
import org.joone.net.*;
import org.joone.util.LearningSwitch;
import org.joone.samples.util.ParityInputSynapse;
import org.joone.io.*;

/**
 * Sample class to demostrate the use of the FahlmanTeacherSynapse and related classes.
 */
public class XORFahlman implements Serializable, NeuralNetListener, NeuralValidationListener {
    
    /** The NN. */
    private NeuralNet nnet = null;
    
    /** FIFO-queue to remember at which cycle the fahlman criterion was fullfilled. */
    private Vector validationCycles = new Vector();
    
    private long mills;
    
    /**
     * @param args the command line arguments
     */
    public static void main(String args[]) {
        XORFahlman xor = new XORFahlman();
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
        
        ParityInputSynapse inputStream = new ParityInputSynapse();
        inputStream.setParitySize(2); // size 2 equals XOR problem
        
        InputConnector myInputData = new InputConnector();
        myInputData.setInputSynapse(inputStream);
        myInputData.setAdvancedColumnSelector("1-2");
        
        InputConnector myInputValData = new InputConnector();
        myInputValData.setInputSynapse(inputStream);
        myInputValData.setAdvancedColumnSelector("1-2");
        
        LearningSwitch mySwitch = new LearningSwitch();
        mySwitch.addTrainingSet(myInputData);
        mySwitch.addValidationSet(myInputValData);
        input.addInputSynapse(mySwitch);
        
        FahlmanTeacherSynapse myFahlman = new FahlmanTeacherSynapse();
        
        InputConnector myDesiredData = new InputConnector();
        myDesiredData.setInputSynapse(inputStream);
        myDesiredData.setAdvancedColumnSelector("3");
        
        InputConnector myDesiredValData = new InputConnector();
        myDesiredValData.setInputSynapse(inputStream);
        myDesiredValData.setAdvancedColumnSelector("3");
        
        LearningSwitch myOutputSwitch = new LearningSwitch();
        myOutputSwitch.addTrainingSet(myDesiredData);
        myOutputSwitch.addValidationSet(myDesiredValData);
        
        TeachingSynapse trainer = new TeachingSynapse(myFahlman);
        trainer.setDesired(myOutputSwitch);
        
        // Connects the Teacher to the last layer of the net
        output.addOutputSynapse(trainer);
        
        nnet = new NeuralNet();
        nnet.addLayer(input, NeuralNet.INPUT_LAYER);
        nnet.addLayer(hidden, NeuralNet.HIDDEN_LAYER);
        nnet.addLayer(output, NeuralNet.OUTPUT_LAYER);
        nnet.setTeacher(trainer);
        nnet.getMonitor().setTrainingPatterns(4);
        nnet.getMonitor().setValidationPatterns(4);
        nnet.getMonitor().setTotCicles(10000);
        nnet.getMonitor().setLearning(true);
        nnet.getMonitor().setLearningRate(0.8);
        nnet.getMonitor().setMomentum(0.3);
        
        mills = System.currentTimeMillis();
        
        nnet.addNeuralNetListener(this);
        nnet.go();
    }
    
    /**
     * Method declaration
     */
    public void netStopped(NeuralNetEvent e) {
        long delay = System.currentTimeMillis() - mills;
        System.out.println("Training finished after "+delay+" ms");
        System.exit(0);
    }
    
    public void cicleTerminated(NeuralNetEvent e) {
        Monitor mon = (Monitor) e.getSource();
        long	c = mon.getCurrentCicle();
        // We validate each 200 epochs
        if (c % 200 == 0) {
            nnet.getMonitor().setExporting(true);
            NeuralNet myClone = nnet.cloneNet();
            nnet.getMonitor().setExporting(false);
            
            myClone.removeAllListeners();
            myClone.getMonitor().setParam(FahlmanTeacherSynapse.CRITERION, Boolean.TRUE);
            
            NeuralNetValidator myValidator = new NeuralNetValidator(myClone);
            myValidator.addValidationListener(this);
            validationCycles.add(new Integer(nnet.getMonitor().getTotCicles() - nnet.getMonitor().getCurrentCicle()));
            myValidator.start();
        }
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
        if (c % 100 == 0) {
            System.out.println(c + " cycles remaining - Error = " + mon.getGlobalError());
        }
    }
    
    public void netStoppedError(NeuralNetEvent e,String error) {
    }
    
    public void netValidated(NeuralValidationEvent event) {
        Monitor myMonitor = ((NeuralNet)event.getSource()).getMonitor();
        
        if(myMonitor.getParam("FAHLMAN_CRITERION") != null &&
                ((Boolean)myMonitor.getParam("FAHLMAN_CRITERION")).booleanValue()) {
            if (nnet.isRunning()) {
                nnet.stop();
                System.out.println("Fahlman criterion fulfilled (at cycle "
                        + ((Integer)validationCycles.get(0)).intValue() + ")...");
            }
        }
        validationCycles.remove(0);
    }
    
}
