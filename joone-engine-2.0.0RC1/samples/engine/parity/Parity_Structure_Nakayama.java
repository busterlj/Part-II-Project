package org.joone.samples.engine.parity;

import java.util.Vector;
import org.joone.engine.*;
import org.joone.engine.learning.*;
import org.joone.engine.listeners.*;
import org.joone.io.*;
import org.joone.log.*;
import org.joone.net.*;
import org.joone.structure.Nakayama;

/**
 * Sample class to show the working of the technique for optimizing activation
 * functions.
 *
 * @author Boris Jansen
 */
public class Parity_Structure_Nakayama implements NeuralNetListener, java.io.Serializable {
    
    /** Logger for this class. */
    private static final ILogger log = LoggerFactory.getLogger(Nakayama.class);
    
    private NeuralNet nnet = null;
    private MemoryInputSynapse inputSynapse, desiredOutputSynapse;
    private MemoryOutputSynapse outputSynapse;
    
    /** Optimizer. */
    private Nakayama nakayama;
    
    // Parity input
    private double[][] inputArray = new double[][] {
        {0.0, 0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0, 1.0},
        {0.0, 0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0, 1.0},
        {0.0, 1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0, 1.0},
        {0.0, 1.0, 1.0, 0.0},
        {0.0, 1.0, 1.0, 1.0},
        {1.0, 0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0, 1.0},
        {1.0, 0.0, 1.0, 0.0},
        {1.0, 0.0, 1.0, 1.0},
        {1.0, 1.0, 0.0, 0.0},
        {1.0, 1.0, 0.0, 1.0},
        {1.0, 1.0, 1.0, 0.0},
        {1.0, 1.0, 1.0, 1.0}
    };
    
    // Parity desired output
    private double[][]	desiredOutputArray = new double[][] {
        {0.0},
        {1.0},
        {1.0},
        {0.0},
        {1.0},
        {0.0},
        {0.0},
        {1.0},
        {1.0},
        {0.0},
        {0.0},
        {1.0},
        {0.0},
        {1.0},
        {1.0},
        {0.0}
    };
    
    /**
     * @param args the command line arguments
     */
    public static void main(String args[]) {
        Parity_Structure_Nakayama parity = new Parity_Structure_Nakayama();
        
        parity.initNeuralNet();
        parity.train();
    }
    
    /**
     * Method declaration
     */
    public void train() {
        // set the inputs
        inputSynapse.setInputArray(inputArray);
        inputSynapse.setAdvancedColumnSelector("1-4");
        
        // set the desired outputs
        desiredOutputSynapse.setInputArray(desiredOutputArray);
        desiredOutputSynapse.setAdvancedColumnSelector("1");
        
        // get the monitor object to train or feed forward
        Monitor monitor = nnet.getMonitor();
        
        // set the monitor parameters
        monitor.setUseRMSE(false);
        monitor.setLearningRate(0.5);
        monitor.setMomentum(0.3);
        monitor.setTrainingPatterns(inputArray.length);
        monitor.setTotCicles(5000);
        monitor.setLearning(true);
        // RPROP parameters
        monitor.addLearner(0, "org.joone.engine.RpropLearner");
        monitor.addLearner(1, "org.joone.engine.BatchLearner");
        monitor.addLearner(2, "org.joone.engine.BasicLearner");
        monitor.setBatchSize(inputArray.length);
        monitor.setLearningMode(2);
        // Nakayama networks work only in multi-thread mode
        monitor.setSingleThreadMode(false);
        nnet.addNeuralNetListener(this);
        nnet.go();
    }
    
    private void test() {
        nnet.getMonitor().setTotCicles(1);
        nnet.getMonitor().setLearning(false);
        // Nakayama networks work only in multi-thread mode
        nnet.getMonitor().setSingleThreadMode(false);
        // Enables the MemoryOutputSynapse in order to get the results of the network
        outputSynapse.setEnabled(true);
        nnet.removeAllListeners();
        nnet.go(true);
        Vector patts = outputSynapse.getAllPatterns();
        System.out.println("\nResults:");
        for(int i = patts.size(); i > 0; i--) {
            Pattern pattern = (Pattern)patts.elementAt(patts.size() - i);
            System.out.println("Output Pattern #"+(patts.size() - i)+" = " + pattern.getArray()[0]);
        }
        System.out.println("Final RMSE: " + nnet.getMonitor().getGlobalError());
    }
    
    /**
     * Method declaration
     */
    protected void initNeuralNet() {
        // First create the three layers
        LinearLayer input = new LinearLayer();
        SigmoidLayer hiddenSigmoid = new SigmoidLayer();
        SineLayer hiddenSine = new SineLayer();
        GaussLayer hiddenGauss = new GaussLayer();
        SigmoidLayer output = new SigmoidLayer();
        
        // set the dimensions of the layers
        input.setRows(4);
        hiddenSigmoid.setRows(8);
        hiddenSine.setRows(8);
        hiddenGauss.setRows(8);
        output.setRows(1);
        
        // Now create the synapses
        FullSynapse synapse_IHSIGMOID = new FullSynapse();
        FullSynapse synapse_IHSINE = new FullSynapse();
        FullSynapse synapse_IHGAUSS = new FullSynapse();
        FullSynapse synapse_HSIGMOIDO = new FullSynapse();
        FullSynapse synapse_HSINEO = new FullSynapse();
        FullSynapse synapse_HGAUSSO = new FullSynapse();
        
        input.addOutputSynapse(synapse_IHSIGMOID);
        input.addOutputSynapse(synapse_IHSINE);
        input.addOutputSynapse(synapse_IHGAUSS);
        
        hiddenSigmoid.addInputSynapse(synapse_IHSIGMOID);
        hiddenSine.addInputSynapse(synapse_IHSINE);
        hiddenGauss.addInputSynapse(synapse_IHGAUSS);
        
        hiddenSigmoid.addOutputSynapse(synapse_HSIGMOIDO);
        hiddenSine.addOutputSynapse(synapse_HSINEO);
        hiddenGauss.addOutputSynapse(synapse_HGAUSSO);
        
        output.addInputSynapse(synapse_HSIGMOIDO);
        output.addInputSynapse(synapse_HSINEO);
        output.addInputSynapse(synapse_HGAUSSO);
        
        // the input to the neural net
        inputSynapse = new MemoryInputSynapse();
        input.addInputSynapse(inputSynapse);
        
        // the output of the neural net
        outputSynapse = new MemoryOutputSynapse();
        output.addOutputSynapse(outputSynapse);
        // NEVER use an enabled xxxOutputSynapse when in training mode
        outputSynapse.setEnabled(false);
        
        // The Trainer and its desired output
        desiredOutputSynapse = new MemoryInputSynapse();
        TeachingSynapse trainer = new TeachingSynapse();
        trainer.setDesired(desiredOutputSynapse);
        
        // Now we add this structure to a NeuralNet object
        nnet = new NeuralNet();
        nnet.addLayer(input, NeuralNet.INPUT_LAYER);
        nnet.addLayer(hiddenSigmoid, NeuralNet.HIDDEN_LAYER);
        nnet.addLayer(hiddenSine, NeuralNet.HIDDEN_LAYER);
        nnet.addLayer(hiddenGauss, NeuralNet.HIDDEN_LAYER);
        nnet.addLayer(output, NeuralNet.OUTPUT_LAYER);
        nnet.setTeacher(trainer);
        output.addOutputSynapse(trainer);
        
        nakayama = new Nakayama(nnet);
        nakayama.addLayer(hiddenSigmoid);
        nakayama.addLayer(hiddenSine);
        nakayama.addLayer(hiddenGauss);
        
//        ErrorBasedConvergenceObserver myObserver = new ErrorBasedConvergenceObserver();
//        myObserver.setPercentage(0.01);
//        myObserver.addConvergenceListener(nakayama);
//        nnet.addNeuralNetListener(myObserver);
        DeltaBasedConvergenceObserver myObserver = new DeltaBasedConvergenceObserver();
        myObserver.setSize(0.0005);
        myObserver.setNeuralNet(nnet);
        myObserver.addConvergenceListener(nakayama);
        nnet.addNeuralNetListener(myObserver);
    }
    
    public void cicleTerminated(NeuralNetEvent e) {
    }
    
    public void errorChanged(NeuralNetEvent e) {
        Monitor mon = (Monitor)e.getSource();
        int c = mon.getTotCicles()-mon.getCurrentCicle();
        if ((c % 100) == 0) {
            System.out.println("Cycle: "+c+" (R)MSE:"+mon.getGlobalError());
        }
        
//        if(mon.getCurrentCicle() == 5000 || mon.getCurrentCicle() == 1000) {
//            nakayama.optimize();
//        }
    }
    
    public void netStarted(NeuralNetEvent e) {
    }
    
    public void netStopped(NeuralNetEvent e) {
        test();
    }
    
    public void netStoppedError(NeuralNetEvent e, String error) {
    }
    
}
