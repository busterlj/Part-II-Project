/*
 * NetTrainer.java
 *
 * Created on 13 ottobre 2003, 19.41
 */

package org.joone.samples.engine.validation;

import java.util.*;
import org.joone.engine.*;
import org.joone.net.*;
/**
 * This class trains and validates a neural network passed as parameter
 * of the constructor and, when the validation phase is finished, it
 * notifies its listeners.
 * The neural network passes as parameter is cloned before to use it,
 * so the calling program can call many copies of this class to train
 * and validate several copies of the same neural network.
 *
 * @author  pmarrone
 */
public class NeuralNetTrainer  implements Runnable, NeuralNetListener, NeuralValidationListener {
    private Vector listeners;
    private NeuralNet nnet;
    private Thread myThread = null;
    
    public NeuralNetTrainer(NeuralNet nn) {
        listeners = new Vector();
        nnet = cloneNet(nn);
    }
    
    public void addValidationListener(NeuralValidationListener newListener){
        if (!listeners.contains(newListener))
            listeners.addElement(newListener);
    }
    
    /**
     * Trains the neural network
     */
    protected void train() {
        //System.out.println("Training...");
        nnet.getMonitor().addNeuralNetListener(this);
        nnet.getMonitor().setLearning(true);
        nnet.getMonitor().setValidation(false);
        nnet.go(true);
        this.validate();
    }
    
    /**
     * Validates the trained neural network
     */
    protected void validate(){
        //System.out.println("Valitading...");
        // Set all the parameters for the validation
        NeuralNet newNet = cloneNet(nnet);
        NeuralNetValidator nnv = new NeuralNetValidator(newNet);
        nnv.addValidationListener(this);
        nnv.start();  // Validates the net
    }
    
    /**
     * Clones the neural network passed as parameter
     */
    private NeuralNet cloneNet(NeuralNet net)  {
        // Creates a copy of the neural network
        net.getMonitor().setExporting(true);
        NeuralNet newNet = net.cloneNet();
        net.getMonitor().setExporting(false);
        
        // Cleans the old listeners
        // This is a fundamental action to avoid that the validated net
        // calls any method of previously registered listeners
        newNet.removeAllListeners();
        return newNet;
    }
    
    // Notifies all the registered listeners
    private void fireNetValidated(NeuralValidationEvent event) {
        NeuralNet NN = (NeuralNet)event.getSource();
        NN.terminate(false); // <-- Added as a bug workaround 
        for (int i=0; i < listeners.size(); ++i) {
            NeuralValidationListener nvl = (NeuralValidationListener)listeners.elementAt(i);
            nvl.netValidated(new NeuralValidationEvent(NN));
        }
    }
    
    /** Starts the training & validation phases into a separated thread
     */
    public void start() {
        if (myThread == null) {
            myThread = new Thread(this, "Trainer");
            myThread.start();
        }
    }
    
    public void run() {
        this.train();
        myThread = null;
    }
    
    public void netStopped(NeuralNetEvent e) {
    }
    
    public void netValidated(NeuralValidationEvent event) {
        // When also the validation phase terminates, then notifies all the listeners
        this.fireNetValidated(event);
    }
    
    public void cicleTerminated(NeuralNetEvent e) {
        //System.out.println("Cycle "+nnet.getMonitor().getCurrentCicle()+" terminated");
    }
    
    public void netStarted(NeuralNetEvent e) {
    }
    
    public void errorChanged(NeuralNetEvent e) {
        /*System.out.println("Error "+nnet.getMonitor().getCurrentCicle()+" changed");
        System.out.println("Tot Cycles: "+nnet.getMonitor().getTotCicles());
        System.out.println("Val. Patt.: "+nnet.getMonitor().getValidationPatterns());
        System.out.println("Tr.  Patt.: "+nnet.getMonitor().getTrainingPatterns()); */
    }
    
    public void netStoppedError(NeuralNetEvent e, String error) {
        System.exit(1);
    }
    
    
}
