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
 * <p>This synapse creates input and desired output data for the parity problem.
 * The parity problem is the problem where numbers (in binary form) having an even
 * number of ones should output a zero and numbers having an odd number of ones
 * should output a one.
 * <p>The partity problem is a problem that is often used to test new learning
 * algorithms, etc., because the problem is quite difficult. Whenever one bit
 * changes the output should be the opposite.
 * <p>This class is also mainly created to test new techniques.
 * <p>The default size of the parity problem is two, that is, it creates the training
 * data for the XOR problem.
 *
 * @author Boris Jansen
 */
public class ParityInputSynapse extends StreamInputSynapse {
    
    /** Logger */
    private static final ILogger log = LoggerFactory.getLogger(ParityInputSynapse.class);
    
    /** The size of the parity problem, that is, the number of bits for the input data. 
     The default is 2, which is just the XOR problem. */
    private int paritySize = 2; // default
    
    /** The value for the upper bit. */
    private double upperBit = 1.0; // default
    
    /** The value for the lower bit. */
    private double lowerBit = 0.0; // default
    
    /** Creates a new instance of ParityInputSynapse */
    public ParityInputSynapse() {
    }
    
    protected void initInputStream() throws JooneRuntimeException {
        setAdvancedColumnSelector("1-" + (paritySize + 1));
        setTokens(new MemoryInputTokenizer(createParityArray()));
        
    }
    
    /**
     * Sets the number of bits the parity problem should consist of, that is, the number
     * of bits the input should consist of. For example, a size 2, is the XOR problem.
     *
     * @param aSize the size / number of bits for the input.
     */
    public void setParitySize(int aSize) {
        paritySize = aSize;
    }
    
    /**
     * Gets the parity size, that is, the number of bits of the input pattern.
     *
     * @return the parity size.
     */
    public int getParitySize() {
        return paritySize;
    }
    
    /**
     * Gets the number of patterns that exist for the parity problem, that is
     * <code>2^parity-size</code>.
     *
     * @return the number of patterns that exist for the parity problem based on the
     * parity size.
     */
    public int getNumOfPatterns() {
        return (int)Math.pow(2, paritySize);
    }
    
    /**
     * Creates an array holding the input and desired output values for the parity problem.
     *
     * @return an array holding an instance of the parity problem.
     */
    protected double[][] createParityArray() {
        int myPatterns = getNumOfPatterns();
        int myDesiredOutput, myBit, myTemp;
        
        double[][] myParityInstance = new double[myPatterns][paritySize + 1];
        for(int i = 0; i < myPatterns; i++) {
            myDesiredOutput = 0;
            myTemp = i;
            for(int j = 0; j < paritySize; j++) {
                myBit = myTemp % 2;
                myTemp = myTemp / 2;
                if(myBit == 1) {
                    myDesiredOutput = (myDesiredOutput + 1) % 2;
                }
                if(myBit == 0) {
                    myParityInstance[i][j] = getLowerBit();
                } else {
                    myParityInstance[i][j] = getUpperBit();
                }
            }
            if(myDesiredOutput == 0) {
                myParityInstance[i][paritySize] = getLowerBit();
            } else {
                myParityInstance[i][paritySize] = getUpperBit();
            }
        }
        
        /* debug 
        for(int i = 0; i < myParityInstance.length; i++) {
            String myText = "";
            for(int j = 0; j < myParityInstance[i].length; j++) {
                myText += myParityInstance[i][j] + " ";                
            }
            log.debug(myText);
        }
        // end debug */
        return myParityInstance;
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
