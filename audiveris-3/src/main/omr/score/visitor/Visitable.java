//----------------------------------------------------------------------------//
//                                                                            //
//                             V i s i t a b l e                              //
//                                                                            //
//----------------------------------------------------------------------------//
// <editor-fold defaultstate="collapsed" desc="hdr">                          //
//  Copyright (C) Herve Bitteur 2000-2010. All rights reserved.               //
//  This software is released under the GNU General Public License.           //
//  Goto http://kenai.com/projects/audiveris to report bugs or suggestions.   //
//----------------------------------------------------------------------------//
// </editor-fold>
package omr.score.visitor;


/**
 * Interface <code>Visitable</code> must be implemented by any node to be
 * visited in the Score hierarchy
 *
 * @author Hervé Bitteur
 */
public interface Visitable
{
    //~ Methods ----------------------------------------------------------------

    //--------//
    // accept //
    //--------//
    /**
     * General entry for any visiting
     *
     * @param visitor concrete visitor object to define the actual processing
     * @return true if children must be visited also, false otherwise.
     *              <b>Nota</b>: Unless there is a compelling reason, it's safer
     *              to return true to let the visitor work normally.
     */
    boolean accept (ScoreVisitor visitor);
}
