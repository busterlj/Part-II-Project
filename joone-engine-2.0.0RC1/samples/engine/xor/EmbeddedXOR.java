/*
 * EmbeddedXOR.java
 *
 * Created on 7 maggio 2002, 19.27
 */

package org.joone.samples.engine.xor;

import org.joone.log.*;
import org.joone.engine.*;
import org.joone.io.*;
import org.joone.net.NeuralNet;

import java.io.*;
import org.joone.util.UnNormalizerOutputPlugIn;
/**
 * This example shows the use of a neural network embedded in another
 * application that gets the output from the MemoryOutputSynapse object
 * querying the neural network with a predefined set of patterns
 *
 * @author  pmarrone
 */
public class EmbeddedXOR {
    private static final ILogger log = LoggerFactory.getLogger(EmbeddedXOR.class);
    private double[][] inputArray = { {0, 0}, {0, 1}, {1, 0}, {1, 1} };
    private MemoryOutputSynapse memOut;
    private static String xorNet = "org/joone/samples/engine/xor/xor.snet";
    /** Creates a new instance of EmbeddedXOR */
    public EmbeddedXOR() {
    }
    
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        EmbeddedXOR xor = new EmbeddedXOR();
        xor.Go(xorNet);
    }
    
    private void Go(String fileName) {
        // We load the serialized XOR neural net
        NeuralNet xor = restoreNeuralNet(fileName);
        if (xor != null) {
            /* We get the first layer of the net (the input layer),
               then remove all the input synapses attached to it
               and attach a MemoryInputSynapse */
            Layer input = xor.getInputLayer();
            input.removeAllInputs();
            MemoryInputSynapse memInp = new MemoryInputSynapse();
            memInp.setFirstRow(1);
            memInp.setAdvancedColumnSelector("1,2");
            
            input.addInputSynapse(memInp);
            memInp.setInputArray(inputArray);
            
            /* We get the last layer of the net (the output layer),
               then remove all the output synapses attached to it
               and attach a MemoryOutputSynapse */
            Layer output = xor.getOutputLayer();
            output.removeAllOutputs();
            memOut = new MemoryOutputSynapse();
            // Inserts an output plugin to set the output range to [1, 2]
            // (just to demonstrate how to use an UnNormalizerPlugin)
            UnNormalizerOutputPlugIn outPlugin = new UnNormalizerOutputPlugIn();
            outPlugin.setAdvancedSerieSelector("1");
            outPlugin.setOutDataMin(1);
            outPlugin.setOutDataMax(2);
            memOut.addPlugIn(outPlugin);
            output.addOutputSynapse(memOut);
            // Now we interrogate the net once with four input patterns
            xor.getMonitor().setTotCicles(1);
            xor.getMonitor().setTrainingPatterns(4);
            xor.getMonitor().setLearning(false);
            interrogate(xor, 10);
            log.info("Finished");
        }
    }
    
    private void interrogate(NeuralNet net, int times) {
        int cc = net.getMonitor().getTrainingPatterns();
        for (int t=0; t < times; ++t) {
            log.info("Launch #"+(t+1));
            net.go();
            for (int i=0; i < cc; ++i) {
                // Read the next pattern and print out it
                double[] pattern = memOut.getNextPattern();
                log.info("    Output Pattern #"+(i+1)+" = "+pattern[0]);
            }
            net.stop();
        }
    }
    
    private NeuralNet restoreNeuralNet(String fileName) {
        NeuralNet nnet = null;
        try {
            FileInputStream stream = new FileInputStream(fileName);
            ObjectInput input = new ObjectInputStream(stream);
            nnet = (NeuralNet)input.readObject();
        } catch (Exception e) {
            log.warn( "Exception was thrown. Message is : " + e.getMessage(),
                    e );
        }
        return nnet;
    }
    
}
