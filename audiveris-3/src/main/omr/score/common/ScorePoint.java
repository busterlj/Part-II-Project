//----------------------------------------------------------------------------//
//                                                                            //
//                            S c o r e P o i n t                             //
//                                                                            //
//----------------------------------------------------------------------------//
// <editor-fold defaultstate="collapsed" desc="hdr">                          //
//  Copyright (C) Herve Bitteur 2000-2010. All rights reserved.               //
//  This software is released under the GNU General Public License.           //
//  Goto http://kenai.com/projects/audiveris to report bugs or suggestions.   //
//----------------------------------------------------------------------------//
// </editor-fold>
package omr.score.common;


/**
 * Class <code>ScorePoint</code> is a simple Point that is meant to represent a
 * point in the Score display, where Systems are arranged horizontally one after
 * the other, and where coordinates are expressed in units.
 *
 * <p>This specialization is used to take benefit of compiler checks, to prevent
 * the use of points with incorrect meaning or units.
 *
 * @author Hervé Bitteur
 */
public class ScorePoint
    extends SimplePoint
{
    //~ Constructors -----------------------------------------------------------

    //------------//
    // ScorePoint //
    //------------//
    /**
     * Creates a new ScorePoint object.
     */
    public ScorePoint ()
    {
    }

    //------------//
    // ScorePoint //
    //------------//
    /**
     * Creates a new ScorePoint object, by cloning an untyped point
     *
     * @param x abscissa
     * @param y ordinate
     */
    public ScorePoint (int x,
                       int y)
    {
        super(x, y);
    }
}
