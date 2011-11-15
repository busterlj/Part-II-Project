//----------------------------------------------------------------------------//
//                                                                            //
//                       S i m p l e D i m e n s i o n                        //
//                                                                            //
//----------------------------------------------------------------------------//
// <editor-fold defaultstate="collapsed" desc="hdr">                          //
//  Copyright (C) Herve Bitteur 2000-2010. All rights reserved.               //
//  This software is released under the GNU General Public License.           //
//  Goto http://kenai.com/projects/audiveris to report bugs or suggestions.   //
//----------------------------------------------------------------------------//
// </editor-fold>
package omr.score.common;

import java.awt.Dimension;

/**
 * Class <code>SimpleDimension</code> is meant to be subclassed to represent a
 * dimension in a specific context.
 *
 * @author Hervé Bitteur
 */
public abstract class SimpleDimension
    extends Dimension
{
    //~ Constructors -----------------------------------------------------------

    //-----------------//
    // SimpleDimension //
    //-----------------//
    /**
     * Creates an instance of <code>SimpleDimension</code> with a width of zero
     * and a height of zero.
     */
    public SimpleDimension ()
    {
    }

    //-----------------//
    // SimpleDimension //
    //-----------------//
    /**
     * Constructs a <code>SimpleDimension</code> and initializes it to the
     * specified width and specified height.
     *
     * @param width the specified width
     * @param height the specified height
     */
    public SimpleDimension (int width,
                            int height)
    {
        super(width, height);
    }

    //~ Methods ----------------------------------------------------------------

    //----------//
    // toString //
    //----------//
    @Override
    public String toString ()
    {
        StringBuilder sb = new StringBuilder(getClass().getSimpleName());
        sb.append("[width=")
          .append(width)
          .append(",height=")
          .append(height)
          .append("]");

        return sb.toString();
    }
}
