/*
 * ValidationSample.java
 *
 * Created on 11 november 2002, 22.59
 * @author  pmarrone
 */

package org.joone.samples.engine.validation;

import java.io.File;
import org.joone.engine.*;
import org.joone.engine.learning.*;
import org.joone.net.*;
import org.joone.io.*;
import org.joone.util.*;

/**
 * This example shows how to check the training level of the net
 * using a validation data source.
 * In this example we will learn to use the following objects:
 * - org.joone.util.LearningSwitch
 * - org.joone.net.NeuralNetValidator
 * - org.joone.util.NormalizerPlugIn
 *
 * This program shows how to build the same kind of neural net as that
 * contained into the org/joone/samples/editor/scripting/ValidationSample.ser
 * file using only java code and the core engine's API. Open that net in
 * the GUI editor to see the architecture of the net built in this example.
 */
public class SimpleValidationSample implements NeuralNetListener, NeuralValidationListener {
    
    NeuralNet net;
    long startms;
    private static String filePath = "org/joone/samples/engine/validation";
    
    /** Creates a new instance of SampleScript */
    public SimpleValidationSample() {
    }
    
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {

        SimpleValidationSample sampleNet = new SimpleValidationSample();
        sampleNet.initialize(filePath);
        sampleNet.start();
    }
    
    private void initialize(String path) {
        /* Creates the three layers and connect them */
        LinearLayer ILayer = new LinearLayer();  // Input Layer
        SigmoidLayer HLayer = new SigmoidLayer(); // Hidden Layer
        SigmoidLayer OLayer = new SigmoidLayer(); // Output Layer
        ILayer.setRows(13); // The input pattern has 13 columns
        HLayer.setRows(4);
        OLayer.setRows(1);  // The desired pattern has 1 column
        FullSynapse synIH = new FullSynapse();
        FullSynapse synHO = new FullSynapse();
        this.connect(ILayer, synIH, HLayer);
        this.connect(HLayer, synHO, OLayer);
        
        /* Creates all the required input data sets */
        FileInputSynapse ITdata = this.createInput(path+"/wine.txt",1,2,14); /* The input training data set */
        FileInputSynapse IVdata = this.createInput(path+"/wine.txt",131,2,14); /* The input validation data set */
        FileInputSynapse DTdata = this.createInput(path+"/wine.txt",1,1,1); /* The desired training data set */
        FileInputSynapse DVdata = this.createInput(path+"/wine.txt",131,1,1); /* The desired validation data set */
        
        /* Creates and attach the input learning switch */
        LearningSwitch Ilsw = this.createSwitch(ITdata, IVdata);
        ILayer.addInputSynapse(Ilsw);
        
        /* Creates and attach the desired learning switch */
        LearningSwitch Dlsw = this.createSwitch(DTdata, DVdata);
        TeachingSynapse ts = new TeachingSynapse(); // The teacher of the net
        ts.setDesired(Dlsw);
        OLayer.addOutputSynapse(ts);
        
        /* Now we put all togheter into a NeuralNet object */
        net = new NeuralNet();
        net.addLayer(ILayer, NeuralNet.INPUT_LAYER);
        net.addLayer(HLayer, NeuralNet.HIDDEN_LAYER);
        net.addLayer(OLayer, NeuralNet.OUTPUT_LAYER);
        net.setTeacher(ts);
        
        /* Sets the Monitor's parameters */
        Monitor mon = net.getMonitor();
        mon.setLearningRate(0.4);
        mon.setMomentum(0.5);
        
        mon.setTrainingPatterns(130);
        mon.setValidationPatterns(48);
        
        mon.setTotCicles(1000);
        mon.setLearning(true);
    }
    
    /** Creates a FileInputSynapse */
    private FileInputSynapse createInput(String name, int firstRow, int firstCol, int lastCol) {
        FileInputSynapse input = new FileInputSynapse();
        input.setInputFile(new File(name));
        input.setFirstRow(firstRow);
        if (firstCol != lastCol)
            input.setAdvancedColumnSelector(firstCol+"-"+lastCol);
        else
            input.setAdvancedColumnSelector(Integer.toString(firstCol));
        
        // We normalize the input data in the range 0 - 1
        NormalizerPlugIn norm = new NormalizerPlugIn();
        if (firstCol != lastCol)
            norm.setAdvancedSerieSelector("1-"+Integer.toString(lastCol-firstCol+1));
        else
            norm.setAdvancedSerieSelector("1");
        norm.setMin(0.1);
        norm.setMax(0.9);
        input.addPlugIn(norm);
        return input;
    }
    
    /** Connects two Layers with a Synapse */
    private void connect(Layer ly1, Synapse syn, Layer ly2) {
        ly1.addOutputSynapse(syn);
        ly2.addInputSynapse(syn);
    }
    
    /* Creates a LearningSwitch and attach to it both the training and
       the desired input synapses */
    private LearningSwitch createSwitch(StreamInputSynapse IT, StreamInputSynapse IV) {
        LearningSwitch lsw = new LearningSwitch();
        lsw.addTrainingSet(IT);
        lsw.addValidationSet(IV);
        return lsw;
    }
    
    private void start() {
        // Registers itself as a listener
        net.getMonitor().addNeuralNetListener(this);
        startms = System.currentTimeMillis();
        net.go();
    }
    
    /* Events */
    public void netValidated(NeuralValidationEvent event) {
        // Shows the RMSE at the end of the cycle
        NeuralNet NN = (NeuralNet)event.getSource();
        System.out.println("    Validation Error: "+NN.getMonitor().getGlobalError());
    }
    
    public void cicleTerminated(NeuralNetEvent e) {
        // Prints out the cycle and the training error
        int cycle = net.getMonitor().getTotCicles() - net.getMonitor().getCurrentCicle()+1;
        if (cycle % 200 == 0) { // We validate the net every 200 cycles
            System.out.println("Cycle #"+cycle);
            System.out.println("    Training Error:   " + net.getMonitor().getGlobalError());
            
            // Creates a copy of the neural network
            net.getMonitor().setExporting(true);
            NeuralNet newNet = net.cloneNet();
            net.getMonitor().setExporting(false);
            
            // Cleans the old listeners
            // This is a fundamental action to avoid that the validating net
            // calls the cicleTerminated method of this class
            newNet.removeAllListeners();
            
            // Set all the parameters for the validation
            NeuralNetValidator nnv = new NeuralNetValidator(newNet);
            nnv.addValidationListener(this);
            nnv.start();  // Validates the net
        }
    }
    
    public void errorChanged(NeuralNetEvent e) {
//        Monitor NN = (Monitor)e.getSource();
//        System.out.println("    Actual training error: "+NN.getGlobalError());
    }
    
    public void netStarted(NeuralNetEvent e) {
    }
    
    public void netStopped(NeuralNetEvent e) {
        System.out.println("Stopped after "+(System.currentTimeMillis()-startms)+" ms");
    }
    
    public void netStoppedError(NeuralNetEvent e,String error) {
    }
    
}
