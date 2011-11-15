package org.joone.samples.engine.validation;

import java.util.*;

import org.joone.engine.*;
import org.joone.inspection.implementations.*;
import org.joone.io.*;
import org.joone.net.*;

/*
 * <p>Title: Effort Estimations Through Neural Networks - Joone Test</p>
 * <p>Description: Stage II - Joone Pilot</p>
 * <p>Copyright: Copyright (c) 2003</p>
 * <p>Company: Motorola</p>
 * @author Sebastian Donatti - AGD013
 * @version 1.0.0
 */

public class NeuralNetTester implements Runnable,NeuralNetListener,NeuralValidationListener{
  // Joone Global Variables
  private Vector listeners;
  private NeuralNet oldNetwork;
  private NeuralNet newNetwork;
  private Thread myThread=null;
  private double lastRSME;
  private int minimaEpochs;
  // JoonePilot Global Variables;
  private int problem;
  private boolean relaunch;
  private static int relaunchNumber;
  private boolean learnCurve; // Added to avoid to print all the intermediate errors when in learning curve mode

  // Launches the train & validation of one neural network - Class Constructor
  public NeuralNetTester(NeuralNet _network,boolean lCurve,int _problem){
    problem=_problem;
    listeners=new Vector();
    oldNetwork=_network;
    newNetwork=cloneNet(_network);
    learnCurve=lCurve;
    lastRSME=0;
    minimaEpochs=0;
    relaunch=false;
  }

  public void addValidationListener(NeuralValidationListener newListener){
    if(!listeners.contains(newListener)){
      listeners.addElement(newListener);
    }
  }

  // Trains the neural network - Class Method
  protected void train(){
    newNetwork.getMonitor().addNeuralNetListener(this);
    newNetwork.getMonitor().setLearning(true);
    newNetwork.getMonitor().setValidation(false);
    newNetwork.start();
    newNetwork.getMonitor().Go();
  }

  // Validates the trained neural network - Class Method
  protected void validate(){
    if(!learnCurve){
      // Update Screen
      System.out.print("\nTraining Finished \n");
      System.out.print("\nValidation Started\n\n");
    }
    // Set all the parameters for the validation
    NeuralNetValidator nnv=new NeuralNetValidator(cloneNet(newNetwork));
    nnv.addValidationListener(this);
    // Validate the network
    nnv.start();
  }

  // Clones the neural network passed as parameter - Class Method
  private NeuralNet cloneNet(NeuralNet oldNet){
    // Create a copy of the neural network
    oldNet.getMonitor().setExporting(true);
    NeuralNet newNet=oldNet.cloneNet();
    oldNet.getMonitor().setExporting(false);
    // Clean the old listeners (This is a fundamental action to avoid that the validated net calls any method of previously registered listeners)
    newNet.removeAllListeners();
    return newNet;
  }

  // Notifies all the registered listeners - Class Method
  private void fireNetValidated(NeuralValidationEvent event){
    NeuralNet NN=(NeuralNet)event.getSource();
    for(int i=0;i<listeners.size();++i){
      NeuralValidationListener nvl=(NeuralValidationListener)listeners.elementAt(i);
      nvl.netValidated(new NeuralValidationEvent(NN));
    }
  }

  // Starts the training & validation phases into a separated thread - Class Method
  public void start(){
    if(myThread==null){
      myThread=new Thread(this);
      myThread.start();
    }
  }

  public void run(){
    this.train();
    myThread=null;
  }

  // Interface Method
  public void netStopped(NeuralNetEvent e){
    // Learning Curve is not selected
    if(!learnCurve){
      // Stopped in error
      if(relaunch){
        // Relaunch the training
        relaunchNumber=relaunchNumber+1;
        relaunch=false;
        minimaEpochs=0;
        newNetwork=cloneNet(oldNetwork);
        // Shuffle the Input
        inputRandomize();
        newNetwork.resetInput();
        // Shuffle the network
        switch(problem){
          case 1:
            newNetwork.randomize(0.7);
            break;
          case 2:
            newNetwork.randomize(0.5);
            break;
          case 3:
            newNetwork.randomize(0.5);
            break;
        }
        this.train();
        System.out.print(" Training Relaunched: "+relaunchNumber+"\n\n");
      }
      // Stopped Correctly
      else{
        // Validate the network when the training phase terminates
        this.validate();
      }
    }
    // Learning Curve is Selected
    else{
      // Stopped in error
      if(relaunch){
        // Relaunch the training
        relaunchNumber=relaunchNumber+1;
        System.out.print("Relaunch Number: "+relaunchNumber+"\n");
        relaunch=false;
        minimaEpochs=0;
        newNetwork=cloneNet(oldNetwork);
        // Shuffle the Input
        inputRandomize();
        newNetwork.resetInput();
        // Shuffle the network
        switch(problem){
          case 1:
            newNetwork.randomize(0.7);
            break;
          case 2:
            newNetwork.randomize(0.5);
            break;
          case 3:
            newNetwork.randomize(0.5);
            break;
        }
        this.train();
      }
      // Stopped Correctly
      else{
        // Validate the network when the training phase terminates
        this.validate();
      }
    }
  }

  public void netValidated(NeuralValidationEvent event){
    // Notify all the listeners when also the validation phase terminates
    this.fireNetValidated(event);
    if(!learnCurve){
      // Update Screen
      System.out.print("\nValidation Finished\n");
    }
  }

  public void cicleTerminated(NeuralNetEvent e){
    if(!learnCurve){
      // Declare & Initialize the local variables
      double currentError=newNetwork.getMonitor().getGlobalError();
      int currentCycle=newNetwork.getMonitor().getCurrentCicle();
      int cycles=newNetwork.getMonitor().getTotCicles();
      int printCycle=currentCycle/10;
      // Print the results every 1000 cycles
      if((printCycle*10)==currentCycle){
        // Declare & Initialize the local variables
        int remaining=cycles-currentCycle;
        int percentage=(remaining*100)/cycles;
        // Update Screen
        System.out.print("RSME: "+currentError+"\n");
      }
    }
  }

  public void netStarted(NeuralNetEvent e){
  }

  public void errorChanged(NeuralNetEvent event){
    Monitor monitor=(Monitor)event.getSource();
    double rmse=monitor.getGlobalError();
    // Stop the network if the Global RSME is lower than the Minimum Expected
    if(rmse<=0.05){
      monitor.Stop();
    }
    // Check if the network has become stuck in local minima
    else{
      if(lastRSME<=(monitor.getGlobalError()+0.0000005)){
        minimaEpochs=minimaEpochs+1;
      }
      else{
        minimaEpochs=0;
      }
    }
    lastRSME=monitor.getGlobalError();
    // Relaunches the training if has becomed stuck in a local minima
    if(minimaEpochs==3000){
      // Stop the training in error
      if(!learnCurve){
        System.out.print("\n\n Training Patterns: "+monitor.getTrainingPatterns()+" - LOCAL MINIMA STUCK \n");
      }
      relaunch=true;
      monitor.Stop();
    }
  }

  /** Randomizes the Network Input Vector - Class Method */
  private void inputRandomize(){
    // Get the input Synapse
    boolean notNullFlag=false;
    Layer inputLayer=newNetwork.getInputLayer();
    Vector inputList=inputLayer.getAllInputs();
    StreamInputSynapse inputSynapse;
    inputSynapse=null;
    // Get the Switch
    for(int i=0;i<inputList.size();i++){
      InputSwitchSynapse inputSwitchTemporal=(InputSwitchSynapse)inputList.elementAt(i);
      if(inputSwitchTemporal.getName().equals("Input Switch Synapse")){
        Vector inputSwitch=inputSwitchTemporal.getAllInputs();
        // Get the Synapse
        for(int j=0;j<inputSwitch.size();j++){
          StreamInputSynapse inputSynapseTemporal=(StreamInputSynapse)inputSwitch.elementAt(i);
          if(inputSynapseTemporal.getName().equals("Learning Input Synapse")){
            inputSynapse=inputSynapseTemporal;
            notNullFlag=true;
          }
        }
      }
    }
    // Get the array if the Synapse has been found
    if(notNullFlag){
      Collection inputCollection;
      inputCollection=inputSynapse.Inspections();
      Iterator temporalIterator=inputCollection.iterator();
      InputsInspection temporalInspection=(InputsInspection)temporalIterator.next();
      Object[][] inputElements=temporalInspection.getComponent();
      // Create the random number
      Random joker=new Random();
      int changerPatternNumberA=joker.nextInt(inputElements.length);
      int changedPatternNumberA=joker.nextInt(inputElements.length);
      int changerPatternNumberB=joker.nextInt(inputElements.length);
      int changedPatternNumberB=joker.nextInt(inputElements.length);
      // Shuffle the array order
      Object[] temporalPatternA=inputElements[changedPatternNumberA];
      Object[] temporalPatternB=inputElements[changedPatternNumberB];
      inputElements[changedPatternNumberA]=inputElements[changerPatternNumberA];
      inputElements[changerPatternNumberA]=temporalPatternA;
      inputElements[changedPatternNumberB]=inputElements[changerPatternNumberB];
      inputElements[changerPatternNumberB]=temporalPatternB;
      // Save the new array
      temporalInspection.setComponent(inputElements);
      inputRandomizeCheck();
    }
  }

  /** Used to Check if the Network Input Vector is randomized - Class Method */
private void inputRandomizeCheck(){
  // Get the input Synapse
  boolean notNullFlagCheck=false;
  Layer inputLayerCheck=newNetwork.getInputLayer();
  Vector inputListCheck=inputLayerCheck.getAllInputs();
  StreamInputSynapse inputSynapseCheck;
  inputSynapseCheck=null;
  // Get the Switch
  for(int i=0;i<inputListCheck.size();i++){
    InputSwitchSynapse inputSwitchTemporalCheck=(InputSwitchSynapse)inputListCheck.elementAt(i);
    if(inputSwitchTemporalCheck.getName().equals("Input Switch Synapse")){
      Vector inputSwitch=inputSwitchTemporalCheck.getAllInputs();
      // Get the Synapse
      for(int j=0;j<inputSwitch.size();j++){
        StreamInputSynapse inputSynapseTemporal=(StreamInputSynapse)inputSwitch.elementAt(i);
        if(inputSynapseTemporal.getName().equals("Learning Input Synapse")){
          inputSynapseCheck=inputSynapseTemporal;
          notNullFlagCheck=true;
        }
      }
    }
  }
  // Get the array if the Synapse has been found
  if(notNullFlagCheck){
    Collection inputCollectionCheck;
    inputCollectionCheck=inputSynapseCheck.Inspections();
    Iterator temporalIteratorCheck=inputCollectionCheck.iterator();
    InputsInspection temporalInspectionCheck=(InputsInspection)temporalIteratorCheck.next();
    Object[][] inputElementsCheck=temporalInspectionCheck.getComponent();
    int javaDebugStop =0;
  }
}

  public void netStoppedError(NeuralNetEvent e,String error){
    // Update the Screen
    System.out.print("Stopped in Error \n");
  }
}
