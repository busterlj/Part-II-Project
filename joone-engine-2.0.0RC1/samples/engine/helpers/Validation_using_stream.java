/*
 * Validation_using_stream.java
 *
 * Created on January 18, 2006, 5:30 PM
 *
 */

package org.joone.samples.engine.helpers;

import java.io.File;
import org.joone.engine.Monitor;
import org.joone.engine.NeuralNetListener;
import org.joone.helpers.factory.JooneTools;
import org.joone.io.FileInputSynapse;
import org.joone.net.NeuralNet;
import org.joone.net.NeuralNetAttributes;
import org.joone.util.NormalizerPlugIn;

/**
 * Example to demonstrate how to use the helpers methods of the JooneTools class
 * with a StreamInputSynapse used as data source.
 * @author P.Marrone
 */
public class Validation_using_stream implements NeuralNetListener {
    
    private static final String fileName = "org/joone/samples/engine/helpers/wine.txt";
    private static final int trainingRows = 150;
    
    private double[][] inputTrain;
    private double[][] desiredTrain;
    private double[][] inputTest;
    private double[][] desiredTest;
    
    /**
     * Creates a new instance of Validation_using_stream 
     */
    public Validation_using_stream() {
    }
    
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        Validation_using_stream me = new Validation_using_stream();
        me.go();
    }
    
    private void go() {
        // Prepare the training and testing data set
        FileInputSynapse fileIn = new FileInputSynapse();
        fileIn.setInputFile(new File(fileName));
        fileIn.setAdvancedColumnSelector("1-14");
        // Input data normalized between -1 and 1
        NormalizerPlugIn normIn = new NormalizerPlugIn();
        normIn.setAdvancedSerieSelector("2-14");
        normIn.setMin(-1);
        normIn.setMax(1);
        fileIn.addPlugIn(normIn);
        // Target data normalized between 0 and 1
        NormalizerPlugIn normOut = new NormalizerPlugIn();
        normOut.setAdvancedSerieSelector("1");
        fileIn.addPlugIn(normOut);
        
        // Extract the training data
        inputTrain = JooneTools.getDataFromStream(fileIn, 1, trainingRows, 2, 14);
        desiredTrain = JooneTools.getDataFromStream(fileIn, 1, trainingRows, 1, 1);
        
        // Extract the test data
        inputTest = JooneTools.getDataFromStream(fileIn, trainingRows+1, 178, 2, 14);
        desiredTest = JooneTools.getDataFromStream(fileIn, trainingRows+1, 178, 1, 1);
        
        int[] nodes = { 13, 4, 1 };
        NeuralNet nnet = JooneTools.create_standard(nodes, JooneTools.LOGISTIC);
        // Set optimal values for learning rate and momentum
        nnet.getMonitor().setLearningRate(0.3);
        nnet.getMonitor().setMomentum(0.5);
//        nnet.getMonitor().setSingleThreadMode(false);
        // Trains the network
        JooneTools.train(nnet, inputTrain, desiredTrain, 
                5000,   // Max # of epochs 
                0.010,  // Stop RMSE
                100,    // Epochs between output reports
                this,   // The listener
                false); // Runs in synch mode
        
        // Gets and prints the final values
        NeuralNetAttributes attrib = nnet.getDescriptor();
        System.out.println("Last training rmse="+attrib.getTrainingError()+
                " at epoch "+attrib.getLastEpoch());
        
        double[][] out = JooneTools.compare(nnet, inputTest, desiredTest);
        System.out.println("Comparion of the last "+out.length+" rows:");
        int cols = out[0].length/2;
        for (int i=0; i < out.length; ++i) {
            System.out.print("\nOutput: ");
            for (int x=0; x < cols; ++x) {
                System.out.print(out[i][x]+" ");
            }
            System.out.print("\tTarget: ");
            for (int x=cols; x < cols*2; ++x) {
                System.out.print(out[i][x]+" ");
            }
        }
    }

    public void cicleTerminated(org.joone.engine.NeuralNetEvent e) {
        // Gets the current values
        Monitor mon = (Monitor)e.getSource();
        int epoch = mon.getTotCicles() - mon.getCurrentCicle() + 1;
        double trainErr = mon.getGlobalError();
        
        // Test a clone of the network
        NeuralNet n = e.getNeuralNet().cloneNet();
        double testErr = JooneTools.test(n, inputTest, desiredTest);
        System.out.println("Epoch "+epoch+":\n\tTraining error="+trainErr+"\n\tValidation error="+testErr);
    }

    public void errorChanged(org.joone.engine.NeuralNetEvent e) {
    }

    public void netStarted(org.joone.engine.NeuralNetEvent e) {
        System.out.println("Training...");
    }

    public void netStopped(org.joone.engine.NeuralNetEvent e) {
        System.out.println("Training stopped.");
    }

    public void netStoppedError(org.joone.engine.NeuralNetEvent e, String error) {
        System.out.println("Training stopped with error "+error);
    }
    
}
