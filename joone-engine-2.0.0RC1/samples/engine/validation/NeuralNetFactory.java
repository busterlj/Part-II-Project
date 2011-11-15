/*
 * NeuralNetFactory.java
 *
 * Created on 1 marzo 2004, 21.21
 */

package org.joone.samples.engine.validation;

import org.joone.engine.*;
import org.joone.io.*;
import org.joone.util.*;

/**
 * <p>Title: Effort Estimations Through Neural Networks</p>
 * <p>Description: Stage II - Joone Pilot</p>
 * <p>Copyright: Copyright (c) 2003</p>
 * <p>Company: Motorola</p>
 * @author Sebastian Donatti - AGD013
 * @version 1.0.0
 */

public class NeuralNetFactory{
  /** Connects two Layers with a Synapse - Class Method
   * @param ly1 Origin Layer
   * @param syn Synapse
   * @param ly2 Ending Layer
   */
  public static void connect(Layer ly1,Synapse syn,Layer ly2){
    ly1.addOutputSynapse(syn);
    ly2.addInputSynapse(syn);
  }

  /** Creates a LearningSwitch and attach to it both the training and the validation input synapses - Class Method
   * @param name The LearningSwitch Name
   * @param IT The Input Synapse (Training)
   * @param IV The Input Synapse (Test)
   * @return LearningSwitch
   */
  public static LearningSwitch createSwitch(String name,StreamInputSynapse IT,StreamInputSynapse IV){
    LearningSwitch lsw = new LearningSwitch();
    lsw.setName(name);
    lsw.addTrainingSet(IT);
    lsw.addValidationSet(IV);
    return lsw;
  }

  /** Creates a MemoryInputSynapse - Class Method
   * @param name The Synapse Name
   * @param inData The Synapse Input Data
   * @param firstRow The First Row
   * @param firstCol The First Column
   * @param lastCol The Last Column
   * @return The MemoryInputSynapse
   */
  public static MemoryInputSynapse createInput(String name,double[][] inData,int firstRow,int firstCol,int lastCol){
    MemoryInputSynapse input = new MemoryInputSynapse();
    input.setName(name);
    input.setInputArray(inData);
    input.setFirstRow(firstRow);
    if (firstCol!=lastCol){
      input.setAdvancedColumnSelector(firstCol+"-"+lastCol);
    }
    else{
      input.setAdvancedColumnSelector(Integer.toString(firstCol));
    }
    return input;
  }
}