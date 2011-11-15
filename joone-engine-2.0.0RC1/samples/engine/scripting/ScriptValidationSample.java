/*
 * ValidationSample.java
 *
 * Created on 11 november 2002, 22.59
 * @author  pmarrone
 */

package org.joone.samples.engine.scripting;

import org.joone.engine.*;
import org.joone.engine.learning.*;
import org.joone.net.*;
import org.joone.io.*;
import org.joone.util.*;
import java.io.*;

/**
 * This example shows how to use the joone's scripting engine to
 * to check the training level of the net using a validation
 * data source.
 * In this example we will learn to use the following objects:
 * - org.joone.util.LearningSwitch
 * - org.joone.util.MacroPlugin
 * - org.joone.util.NormalizerPlugIn
 *
 * This program shows how to build the same neural net contained into the
 * org/joone/samples/editor/scripting/ValidationSample.ser file using
 * only java code and the core engine's API. Open that net in
 * the GUI editor to see the architecture of the net built in this example.
 */
public class ScriptValidationSample {
    
    NeuralNet net;
    private static String filePath = "org/joone/samples/engine/scripting";
    /** Creates a new instance of SampleScript */
    public ScriptValidationSample() {
    }
    
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {

        ScriptValidationSample sampleNet = new ScriptValidationSample();
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
        
        /* Reads and inserts the text of the validation macro */
        MacroPlugin mPlugin = new MacroPlugin();
        String validation = this.readFile(new File(path+"/validation.bsh"));
        mPlugin.getMacroManager().addMacro("cycleTerminated", validation);
        /* Sets the scripting */
        mPlugin.setRate(100); // Prints out the results each 100 cycles
        net.setMacroPlugin(mPlugin);
        net.setScriptingEnabled(true); // Enables the scripts' execution
        
        /* Sets the Monitor's parameters */
        Monitor mon = net.getMonitor();
        mon.setLearningRate(0.2);
        mon.setMomentum(0.3);
        
        /* Here we set at the beginning the number of the training rows,
         * because the number of the validation rows is set by the script 
         * at the validation time */
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
    
    /**
     * Reads the content of the file.
     *
     * @param	p_file	the content of the file that is to be read.
     * @return	the content of the file.
     */
    private String readFile(File p_file) 
    {
        String str = null, msg;
        FileReader reader = null;
        
        try 
        {
            reader = new FileReader(p_file);
        
            int tch = new Long(p_file.length()).intValue();
            char[] m_buf = new char[tch];
            int nch = reader.read(m_buf);
            if (nch != -1)
                str = new String(m_buf, 0, nch);
            reader.close();

        } catch (FileNotFoundException fnfe) 
        {
            System.err.println(fnfe.getMessage());
            return str;
        } catch (IOException ioe) 
        {
            System.err.println(ioe.getMessage());
        }
        return str;
    }
    
    private void start() {
        net.go();
    }
}
