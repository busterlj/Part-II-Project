//----------------------------------------------------------------------------//
//                                                                            //
//                               M o m e n t s                                //
//                                                                            //
//----------------------------------------------------------------------------//
// <editor-fold defaultstate="collapsed" desc="hdr">                          //
//  Copyright (C) Herve Bitteur 2000-2010. All rights reserved.               //
//  This software is released under the GNU General Public License.           //
//  Goto http://kenai.com/projects/audiveris to report bugs or suggestions.   //
//----------------------------------------------------------------------------//
// </editor-fold>
package omr.math;

import omr.log.Logger;

import omr.score.common.PixelPoint;

/**
 * Class <code>Moments</code> encapsulates the set of all moments that
 * characterize a figure made of points.
 *
 * @author Hervé Bitteur
 */
public class Moments
{
    //~ Static fields/initializers ---------------------------------------------

    /** Usual logger utility */
    private static final Logger logger = Logger.getLogger(Moments.class);

    /**  Number of moments handled : {@value}*/
    public static final int size = 19;

    /** Labels for better display */
    private static final String[] labels = {
                                               /*  0 */ "weight", /*  1 */
    "width", /*  2 */
    "height", /*  3 */
    "n20", /*  4 */
    "n11", /*  5 */
    "n02", /*  6 */
    "n30", /*  7 */
    "n21", /*  8 */
    "n12", /*  9 */
    "n03", /* 10 */
    "h1", /* 11 */
    "h2", /* 12 */
    "h3", /* 13 */
    "h4", /* 14 */
    "h5", /* 15 */
    "h6", /* 16 */
    "h7", /* 17 */
    "xBar", /* 18 */
    "yBar"
                                           };

    //~ Instance fields --------------------------------------------------------

    /**
     * The various moments, implemented as an array of double's
     */
    private Double[] k = new Double[size];

    //~ Constructors -----------------------------------------------------------

    //---------//
    // Moments //
    //---------//
    /**
     * Default constructor, needed for XML binder
     */
    public Moments ()
    {
    }

    //---------//
    // Moments //
    //---------//
    /**
     * Compute the moments for a set of points whose x and y coordinates are
     * provided, all values being normed by the provided unit value.
     *
     * @param x    the array of abscissa values
     * @param y    the array of ordinate values
     * @param dim  the number of points
     * @param unit the length (number of pixels, for example 20) of norming unit
     */
    public Moments (int[] x,
                    int[] y,
                    int   dim,
                    int   unit)
    {
        // Safety check
        if (unit == 0) {
            ///logger.warning("Cannot compute moments with unit set to 0");
            throw new RuntimeException("Zero-valued unit");
        }

        int    xMin = Integer.MAX_VALUE;
        int    xMax = Integer.MIN_VALUE;
        int    yMin = Integer.MAX_VALUE;
        int    yMax = Integer.MIN_VALUE;

        double dx;
        double dy;

        // Normalized Moments
        double n00 = (double) dim / (double) (unit * unit);
        double n01 = 0d;
        double n02 = 0d;
        double n03 = 0d;
        double n10 = 0d;
        double n11 = 0d;
        double n12 = 0d;
        double n20 = 0d;
        double n21 = 0d;
        double n30 = 0d;

        // Total weight
        double w = dim; // For p+q = 0
        double w2 = w * w; // For p+q = 2
        double w3 = Math.sqrt(w * w * w * w * w); // For p+q = 3

        // Mean x & y
        for (int i = dim - 1; i >= 0; i--) {
            n10 += x[i];
            n01 += y[i];
        }

        n10 /= dim;
        n01 /= dim;

        // width & height
        for (int i = dim - 1; i >= 0; i--) {
            int xx = x[i];

            if (xx < xMin) {
                xMin = xx;
            }

            if (xx > xMax) {
                xMax = xx;
            }

            int yy = y[i];

            if (yy < yMin) {
                yMin = yy;
            }

            if (yy > yMax) {
                yMax = yy;
            }
        }

        for (int i = dim - 1; i >= 0; i--) {
            dx = x[i] - n10;
            dy = y[i] - n01;
            n11 += (dx * dy);
            n12 += (dx * dy * dy);
            n21 += (dx * dx * dy);
            n20 += (dx * dx);
            n02 += (dy * dy);
            n30 += (dx * dx * dx);
            n03 += (dy * dy * dy);
        }

        // Normalize
        //
        // p + q = 2
        n11 /= w2;
        n20 /= w2;
        n02 /= w2;
        //
        // p + q = 3
        n12 /= w3;
        n21 /= w3;
        n30 /= w3;
        n03 /= w3;

        // Assign non-orthogonal centralized moments
        // (invariant to translation & scaling)
        k[0] = n00; // Weight
        k[1] = (double) (xMax - xMin + 1) / unit; // Width
        k[2] = (double) (yMax - yMin + 1) / unit; // Height
        k[3] = n20; // X absolute eccentricity
        k[4] = n11; // XY covariance
        k[5] = n02; // Y absolute eccentricity
        k[6] = n30; // X signed eccentricity
        k[7] = n21; //
        k[8] = n12; //
        k[9] = n03; // Y signed eccentricity
                    //
                    // Assign orthogonals moments (Hu set)
                    // (Invariant to translation / scaling / rotation)

        k[10] = n20 + n02;
        //
        k[11] = ((n20 - n02) * (n20 - n02)) + (4 * n11 * n11);
        //
        k[12] = ((n30 - (3 * n12)) * (n30 - (3 * n12))) +
                ((n03 - (3 * n21)) * (n03 - (3 * n21)));
        //
        k[13] = ((n30 + n12) * (n30 + n12)) + ((n03 + n21) * (n03 + n21));
        //
        k[14] = ((n30 - (3 * n12)) * (n30 + n12) * (((n30 + n12) * (n30 + n12)) -
                                                   (3 * (n21 + n03) * (n21 +
                                                                      n03)))) +
                ((n03 - (3 * n21)) * (n03 + n21) * (((n03 + n21) * (n03 + n21)) -
                                                   (3 * (n12 + n30) * (n12 +
                                                                      n30))));
        //
        k[15] = ((n20 - n02) * (((n30 + n12) * (n30 + n12)) -
                               ((n03 + n21) * (n03 + n21)))) +
                (4 * n11 * (n30 + n12) * (n03 + n21));
        //
        k[16] = (((3 * n21) - n03) * (n30 + n12) * (((n30 + n12) * (n30 + n12)) -
                                                   (3 * (n21 + n03) * (n21 +
                                                                      n03)))) -
                (((3 * n12) - n30) * (n03 + n21) * (((n03 + n21) * (n03 + n21)) -
                                                   (3 * (n12 + n30) * (n12 +
                                                                      n30))));

        // Mass center placed here
        k[17] = n10; // xBar
        k[18] = n01; // yBar
    }

    //~ Methods ----------------------------------------------------------------

    //-------------//
    // getCentroid //
    //-------------//
    /**
     * Report the mass center of the glyph
     *
     * @return the centroid
     */
    public PixelPoint getCentroid ()
    {
        return new PixelPoint((int) Math.rint(k[17]), (int) Math.rint(k[18]));
    }

    //-----------//
    // getHeight //
    //-----------//
    /**
     * Report the height of the glyph, normalized by unit
     *
     * @return the normalized height
     */
    public Double getHeight ()
    {
        return k[2];
    }

    //----------//
    // getLabel //
    //----------//
    /**
     * Report the label related to moment at specified index
     *
     * @param index the moment index
     * @return the related index
     */
    public static String getLabel (int index)
    {
        return labels[index];
    }

    //-----------//
    // getValues //
    //-----------//
    /**
     * Report the array of moment values
     *
     * @return the moment values
     */
    public Double[] getValues ()
    {
        return k;
    }

    //-----------//
    // getWeight //
    //-----------//
    /**
     * Report the total weight of the glyph, normalized by unit**2
     *
     * @return the normalized weight
     */
    public Double getWeight ()
    {
        return k[0];
    }

    //----------//
    // getWidth //
    //----------//
    /**
     * Report the width of the glyph, normalized by unit
     *
     * @return the normalized width
     */
    public Double getWidth ()
    {
        return k[1];
    }

    //----------//
    // toString //
    //----------//
    /**
     * Report a string description of the moments
     *
     * @return a string of description
     */
    @Override
    public String toString ()
    {
        StringBuilder sb = new StringBuilder();
        sb.append("{Moments");

        for (int i = 0; i < k.length; i++) {
            sb.append(" ")
              .append(i)
              .append("/")
              .append(labels[i])
              .append("=")
              .append(String.format("%g", k[i]));
        }

        sb.append("}");

        return sb.toString();
    }
}
