package org.joone.samples.engine.xor.rbf;

import org.joone.engine.*;
import org.joone.engine.learning.*;
import org.joone.io.*;
import org.joone.net.*;
import java.util.Vector;

/**
 * Very simple example of the Gaussian (static and random centers) RBF solving the XOR problem.
 *
 * @author Boris Jansen
 */
public class XOR_static_RBF implements NeuralNetListener {
    
    /** The neural network. */
    private NeuralNet nnet = null;
    
    /** The RBF hidden layer. */
    RbfGaussianLayer hidden = null;
    
    /** Synapses. */
    private MemoryInputSynapse inputSynapse, desiredOutputSynapse;
    private MemoryOutputSynapse outputSynapse;
    
    /** If the following flag is true, the centers will be chosen randomly.
     * Otherwise it will use predefined, fixed centers (able to solve the XOR
     * problem.
     */
    private boolean randomCenters = false;
    
    // XOR input
    private double[][] inputArray = new double[][] {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };
    
    // XOR desired output
    private double[][] desiredOutputArray = new double[][] {
        {1.0},
        {0.0},
        {0.0},
        {1.0}
    };
    
    /**
     * Main
     *
     * @param args the command line arguments
     */
    public static void main(String args[]) {
        XOR_static_RBF xor = new XOR_static_RBF();
        
        xor.initNeuralNet();
        xor.train();
        xor.test();
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
        monitor.setLearningRate(0.3);
        monitor.setMomentum(0.8);
        monitor.setTrainingPatterns(inputArray.length);
        monitor.setTotCicles(200);
        
        // RPROP parameters (uncomment if you want to use the RPROP learning algorithm)
        //monitor.getLearners().add(0, "org.joone.engine.RpropLearner");
        //monitor.setBatchSize(4);
        //monitor.setLearningMode(0);
        
        monitor.setLearning(true);
        nnet.addNeuralNetListener(this);
        nnet.go(true);
    }
    
    /**
     * Create and init the neural network.
     */
    protected void initNeuralNet() {
        // First create the three layers
        LinearLayer input = new LinearLayer();
        hidden = new RbfGaussianLayer();
        //SigmoidLayer output = new SigmoidLayer(); // you can try it (not a traditional RBF network)
        BiasedLinearLayer output = new BiasedLinearLayer();
        
        // set the dimensions of the layers
        input.setRows(2);
        hidden.setRows(2);
        output.setRows(1);
        
        if(!randomCenters) {
            // Use static Gaussian RBFs
            RbfGaussianParameters[] myParameters = new RbfGaussianParameters[2];
            double[] myMean0 = {0.0, 0.0};
            myParameters[0] = new RbfGaussianParameters(myMean0, Math.sqrt(.5));
            double[] myMean1 = {1.0, 1.0};
            myParameters[1] = new RbfGaussianParameters(myMean1, Math.sqrt(.5));
            hidden.setGaussianParameters(myParameters);
        }
        
        // Now create the two synapses
        RbfInputSynapse synapse_IH = new RbfInputSynapse(); /* input -> hidden conn. */
        FullSynapse synapse_HO = new FullSynapse(); /* hidden -> output conn. */
        
        // Connect the input layer whit the hidden layer
        input.addOutputSynapse(synapse_IH);
        hidden.addInputSynapse(synapse_IH);
        
        // Connect the hidden layer whit the output layer
        hidden.addOutputSynapse(synapse_HO);
        output.addInputSynapse(synapse_HO);
        
        // the input to the neural net
        inputSynapse = new MemoryInputSynapse();
        input.addInputSynapse(inputSynapse);
        if(randomCenters) {
            hidden.useRandomCenter(inputSynapse);
        }
        
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
    
    public void test() {
        // attach a MemoryOutputSynapse to the output of the neural net
        outputSynapse = new MemoryOutputSynapse();
        nnet.getOutputLayer().addOutputSynapse(outputSynapse);
        nnet.getMonitor().setTotCicles(1);
        nnet.getMonitor().setTrainingPatterns(4);
        nnet.getMonitor().setLearning(false);
        nnet.removeAllListeners();
        nnet.go();
        
        System.out.println("Outputs");
        System.out.println("-------");
        for(int i = 0; i < 4; i++) {
            double[] myPattern = outputSynapse.getNextPattern();
            System.out.println("Output: " + myPattern[0]);
        }
        
        System.out.println("Centers RBF neurons: ");
        RbfGaussianParameters[] myParams = hidden.getGaussianParameters();
        for(int i = 0; i < myParams.length; i++) {
            String myText = (i+1) + ": [center: ";
            for(int j = 0; j < myParams[i].getMean().length; j++) {
                myText += myParams[i].getMean()[j] + ", ";
            }
            myText += "Std dev: " + myParams[i].getStdDeviation() + "]";
            System.out.println(myText);
        }
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
    }
    
    public void netStoppedError(NeuralNetEvent e, String error) {
    }
    
}
