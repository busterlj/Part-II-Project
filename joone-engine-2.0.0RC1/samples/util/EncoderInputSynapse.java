/*
 * ParityInputSynapse.java
 *
 * Created on November 28, 2004, 4:24 PM
 */

package org.joone.samples.util;

import org.joone.exception.JooneRuntimeException;
import org.joone.io.*;
import org.joone.log.*;


/**
 * <p>This synapse creates input data (which is also the desired output data) for 
 * the Encoder problem.
 * The encoder problem is the problem consisting of N patterns (each pattern also 
 * consists of N bits). Each pattern has only one bit turned on (1) and all the other
 * bits are off (0). The task is to duplicate the input units into the output units, 
 * going through (usually a smaller) hidden layer.
 * 
 * <p>This class is also mainly created to test new techniques.
 *
 * @author Boris Jansen
 */
public class EncoderInputSynapse extends StreamInputSynapse {
    
    /** Logger */
    private static final ILogger log = LoggerFactory.getLogger(EncoderInputSynapse.class);
    
    /** The size of the encoder problem, that is, number of patterns and also at the same
     * time the number of bits each pattern consists of. */
    private int size = 0; // default
    
    /** The value for the upper bit. */
    private double upperBit = 1.0; // default
    
    /** The value for the lower bit. */
    private double lowerBit = 0.0; // default
    
    /** Creates a new instance of EncoderInputSynapse. */
    public EncoderInputSynapse() {
    }
    
    /** Creates a new instance of EncoderInputSynapse. */
    public EncoderInputSynapse(double aLowerBit, double anUpperBit) {
        lowerBit = aLowerBit;
        upperBit = anUpperBit;
    }
    
    protected void initInputStream() throws JooneRuntimeException {
        setAdvancedColumnSelector("1-" + size);
        setTokens(new MemoryInputTokenizer(createEncoderArray()));
    }
    
    /**
     * Sets the number of bits for each pattern. This value is also at the same time the number
     * of patterns.
     *
     * @param aSize the size, number of bits for each pattern and also the number of patterns itself.
     */
    public void setSize(int aSize) {
        size = aSize;
    }
    
    /**
     * Gets the size, that is, the number of bits of the input pattern and at the same time 
     * the number of patterns itself.
     *
     * @return the size.
     */
    public int getSize() {
        return size;
    }
    
    /**
     * Gets the number of patterns that exist for the parity problem.
     *
     * @return the number of patterns that exist.
     * @see {@link:getSize()}
     */
    public int getNumOfPatterns() {
        return size;
    }
    
    /**
     * Creates an array holding the input (which are also the desired output patterns) 
     * patterns for the encoder problem.
     *
     * @return an array holding an instance of the encoder problem.
     */
    protected double[][] createEncoderArray() {
        int myPatterns = getNumOfPatterns();
        
        double[][] myInstance = new double[myPatterns][getSize()];
        for(int i = 0; i < myPatterns; i++) {
            for(int j = 0; j < getSize(); j++) {
                myInstance[i][j] = getLowerBit();
            }
            myInstance[i][i] = getUpperBit();
        }
        
        /* debug 
        for(int i = 0; i < myInstance.length; i++) {
            String myText = "";
            for(int j = 0; j < myInstance[i].length; j++) {
                myText += myInstance[i][j] + " ";                
            }
            log.debug(myText);
        }
        end debug */
        return myInstance;
    }
    
    /**
     * Sets the value for the upper bit.
     *
     * @param aValue the value to use for the upper bit.
     */
    public void setUpperBit(double aValue) {
        upperBit = aValue;
    }
    
    /**
     * Gets the value used for the upper bit.
     *
     * @returns the value used for the upper bit.
     */
    public double getUpperBit() {
        return upperBit;
    }
    
    /**
     * Sets the value for the lower bit. 
     *
     * @param aValue the value to use for the lower bit.
     */
    public void setLowerBit(double aValue) {
        lowerBit = aValue;
    }
    
    /**
     * Gets the value used for the lower bit.
     *
     * @returns the value used for the lower bit.
     */
    public double getLowerBit() {
        return lowerBit;
    }
}
