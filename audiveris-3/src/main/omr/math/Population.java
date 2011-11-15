//----------------------------------------------------------------------------//
//                                                                            //
//                            P o p u l a t i o n                             //
//                                                                            //
//----------------------------------------------------------------------------//
// <editor-fold defaultstate="collapsed" desc="hdr">                          //
//  Copyright (C) Herve Bitteur 2000-2010. All rights reserved.               //
//  This software is released under the GNU General Public License.           //
//  Goto http://kenai.com/projects/audiveris to report bugs or suggestions.   //
//----------------------------------------------------------------------------//
// </editor-fold>
package omr.math;


/**
 * Class <code>Population</code> is used to cumulate measurements, and compute
 * mean value, standard deviation and variance on them.
 *
 * @author Hervé Bitteur
 */
public class Population
{
    //~ Instance fields --------------------------------------------------------

    /** Sum of measured values */
    private double s = 0d;

    /** Sum of squared measured values */
    private double s2 = 0d;

    /** Number of measurements */
    private int n = 0;

    //~ Constructors -----------------------------------------------------------

    //------------//
    // Population //
    //------------//
    /**
     * Construct a structure to cumulate the measured values
     */
    public Population ()
    {
    }

    //~ Methods ----------------------------------------------------------------

    //----------------//
    // getCardinality //
    //----------------//
    /**
     * Get the number of cumulated measurements
     *
     * @return this number
     */
    public int getCardinality ()
    {
        return n;
    }

    //--------------//
    // getMeanValue //
    //--------------//
    /**
     * Retrieve the mean value from the measurements cumulated so far
     *
     * @return the mean value
     */
    public double getMeanValue ()
    {
        if (n == 0) {
            throw new RuntimeException("Population is empty");
        }

        return s / (double) n;
    }

    //----------------------//
    // getStandardDeviation //
    //----------------------//
    /**
     * Get the standard deviation around the mean value
     *
     * @return the standard deviation
     */
    public double getStandardDeviation ()
    {
        return Math.sqrt(getVariance());
    }

    //-------------//
    // getVariance //
    //-------------//
    /**
     * Get the variance around the mean value
     *
     * @return the variance (square of standard deviation)
     */
    public double getVariance ()
    {
        if (n < 2) {
            throw new RuntimeException("Not enough cumulated values : " + n);
        }

        return Math.max(0d, (s2 - ((s * s) / (double) n)) / (double) (n - 1));
    }

    //--------------//
    // excludeValue //
    //--------------//
    /**
     * Remove a measurement from the cumulated values
     *
     * @param val the measure value to remove
     */
    public void excludeValue (double val)
    {
        if (n < 1) {
            throw new RuntimeException("Population is empty");
        }

        n -= 1;
        s -= val;
        s2 -= (val * val);
    }

    //--------------//
    // includeValue //
    //--------------//
    /**
     * Add a measurement to the cumulated values
     *
     * @param val the measure value
     */
    public void includeValue (double val)
    {
        n += 1;
        s += val;
        s2 += (val * val);
    }

    //-------------------//
    // includePopulation //
    //-------------------//
    /**
     * Add a whole population to this one
     * @param other the other population to include
     */
    public void includePopulation (Population other)
    {
        n += other.n;
        s += other.s;
        s2 += other.s2;
    }

    //-------//
    // reset //
    //-------//
    /**
     * Forget all measurements made so far.
     */
    public void reset ()
    {
        n = 0;
        s = 0d;
        s2 = 0d;
    }

    //-------//
    // reset //
    //-------//
    /**
     * Reset to the single measurement provided
     *
     * @param val the new first measured value
     */
    public void reset (double val)
    {
        reset();
        includeValue(val);
    }
}
