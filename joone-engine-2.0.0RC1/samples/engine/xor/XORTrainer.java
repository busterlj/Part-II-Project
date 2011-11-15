/*
 * XORTrainer.java
 *
 * Created on 3 marzo 2004, 17.26
 */

package org.joone.samples.engine.xor;

import org.joone.engine.*;
import org.joone.net.*;
import org.joone.io.*;

import java.io.*;
import java.util.*;

/**
 * This sample application loads a serialized network and launches it in training mode
 * @author  P.Marrone
 */
public class XORTrainer implements NeuralNetListener {
    private static String xorNet = "org/joone/samples/engine/xor/xor.snet";
    
    /** Creates a new instance of XORTrainer */
    public XORTrainer() {
    }
    
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        XORTrainer xor = new XORTrainer();
        xor.Go(xorNet);
    }
    
    private void Go(String fileName) {
        NeuralNet xor = restoreNeuralNet(fileName);
        if (xor != null) {
            xor.getMonitor().addNeuralNetListener(this);
            xor.getMonitor().setLearning(true);
            TreeSet tree = xor.check();
            if (tree.isEmpty()) {
                xor.go(true);
                System.out.println("Network stopped. Last RMSE="+xor.getMonitor().getGlobalError());
            } else {
                Iterator it = tree.iterator();
                while (it.hasNext()) {
                    NetCheck nc = (NetCheck)it.next();
                    System.out.println(nc.toString());
                }
            }
            
        }
    }
    
    private NeuralNet restoreNeuralNet(String fileName) {
        NeuralNet nnet = null;
        try {
            FileInputStream stream = new FileInputStream(fileName);
            ObjectInput input = new ObjectInputStream(stream);
            nnet = (NeuralNet)input.readObject();
        } catch (Exception e) {
            System.out.println( "Exception was thrown. Message is : " + e.getMessage());
        }
        return nnet;
    }
    
    public void cicleTerminated(NeuralNetEvent e) {
    }
    
    public void errorChanged(NeuralNetEvent e) {
        Monitor mon = (Monitor) e.getSource();
        long	c = mon.getCurrentCicle();
        
        // We want to print the results every 200 cycles
        if (c % 200 == 0) {
            System.out.println(c + " epochs remaining - RMSE = " + mon.getGlobalError());
        }
    }
    
    public void netStarted(NeuralNetEvent e) {
        System.out.println("Started...");
    }
    
    public void netStopped(NeuralNetEvent e) {
        System.out.println("Stopped...");
    }
    
    public void netStoppedError(NeuralNetEvent e, String error) {
        System.out.println("Error: "+error);
        System.exit(1);
    }
    
}
