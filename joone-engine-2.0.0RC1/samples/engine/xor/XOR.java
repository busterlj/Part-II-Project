/*
 * XOR.java
 * Sample class to demostrate the use of the Joone's core engine
 * see the Developer Guide for more details
 */

/*
 * JOONE - Java Object Oriented Neural Engine
 * http://joone.sourceforge.net
 */
package org.joone.samples.engine.xor;

import java.io.File;
import org.joone.engine.*;
import org.joone.engine.learning.*;
import org.joone.io.*;
import org.joone.net.NeuralNet;

public class XOR implements NeuralNetListener {
    private static String inputData = "org/joone/samples/engine/xor/xor.txt";
    private static String outputFile = "/tmp/xorout.txt";
    
    /** Creates new XOR */
    public XOR() {
    }
    
    /**
     * @param args the command line arguments
     */
    
    public static void main(String args[]) {
        XOR xor = new XOR();
        xor.Go(inputData, outputFile);
    }
    
    public void Go(String inputFile, String outputFile) {
        /*
         * Firts, creates the three Layers
         */
        LinearLayer input = new LinearLayer();
        SigmoidLayer hidden = new SigmoidLayer();
        SigmoidLayer output = new SigmoidLayer();
        input.setLayerName("input");
        hidden.setLayerName("hidden");
        output.setLayerName("output");
        /* sets their dimensions */
        input.setRows(2);
        hidden.setRows(3);
        output.setRows(1);
        
        /*
         * Now create the two Synapses
         */
        FullSynapse synapse_IH = new FullSynapse(); /* input -> hidden conn. */
        FullSynapse synapse_HO = new FullSynapse(); /* hidden -> output conn. */
        
        synapse_IH.setName("IH");
        synapse_HO.setName("HO");
        /*
         * Connect the input layer whit the hidden layer
         */
        input.addOutputSynapse(synapse_IH);
        hidden.addInputSynapse(synapse_IH);
        /*
         * Connect the hidden layer whit the output layer
         */
        hidden.addOutputSynapse(synapse_HO);
        output.addInputSynapse(synapse_HO);
        
        FileInputSynapse inputStream = new FileInputSynapse();
        /* The first two columns contain the input values */
        inputStream.setAdvancedColumnSelector("1,2");
        
        /* This is the file that contains the input data */
        inputStream.setInputFile(new File(inputFile));
        input.addInputSynapse(inputStream);
        
        
        TeachingSynapse trainer = new TeachingSynapse();
        
        /* Setting of the file containing the desired responses,
         provided by a FileInputSynapse */
        FileInputSynapse samples = new FileInputSynapse();
        samples.setInputFile(new File(inputFile));
        /* The output values are on the third column of the file */
        samples.setAdvancedColumnSelector("3");
        
        trainer.setDesired(samples);
        
        /* Creates the error output file */
        FileOutputSynapse error = new FileOutputSynapse();
        error.setFileName(outputFile);
        //error.setBuffered(false);
        trainer.addResultSynapse(error);
        
        /* Connects the Teacher to the last layer of the net */
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
        
        /* The application registers itself as monitor's listener
         * so it can receive the notifications of termination from
         * the net.
         */
        monitor.addNeuralNetListener(this);
        
        monitor.setTrainingPatterns(4); /* # of rows (patterns) contained in the input file */
        monitor.setTotCicles(2000); /* How many times the net must be trained on the input patterns */
        monitor.setLearning(true); /* The net must be trained */
        nnet.go(); /* The net starts the training job */
    }
    
    public void netStopped(NeuralNetEvent e) {
        System.out.println("Training finished");
    }
    
    public void cicleTerminated(NeuralNetEvent e) {
    }
    
    public void netStarted(NeuralNetEvent e) {
        System.out.println("Training...");
    }
    
    public void errorChanged(NeuralNetEvent e) {
        Monitor mon = (Monitor)e.getSource();
        /* We want print the results every 200 cycles */
        if (mon.getCurrentCicle() % 200 == 0)
            System.out.println(mon.getCurrentCicle() + " epochs remaining - RMSE = " + mon.getGlobalError());
    }
    
    public void netStoppedError(NeuralNetEvent e,String error) {
    }
}
