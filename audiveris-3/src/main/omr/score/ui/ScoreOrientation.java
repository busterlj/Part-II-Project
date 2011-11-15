//----------------------------------------------------------------------------//
//                                                                            //
//                      S c o r e O r i e n t a t i o n                       //
//                                                                            //
//----------------------------------------------------------------------------//
// <editor-fold defaultstate="collapsed" desc="hdr">                          //
//  Copyright (C) Herve Bitteur 2000-2010. All rights reserved.               //
//  This software is released under the GNU General Public License.           //
//  Goto http://kenai.com/projects/audiveris to report bugs or suggestions.   //
//----------------------------------------------------------------------------//
// </editor-fold>
package omr.score.ui;


/**
 * Class <code>ScoreOrientation</code> defines the orientation used for systems
 * layout in the score view
 *
 * @author Hervé Bitteur
 */
public enum ScoreOrientation {
    /** Systems displayed side by side */
    HORIZONTAL("Horizontal"),
    /** System displayed one above the other */
    VERTICAL("Vertical");
    //
    public final String description;

    //------------------//
    // ScoreOrientation //
    //------------------//
    private ScoreOrientation (String description)
    {
        this.description = description;
    }

    //----------//
    // toString //
    //----------//
    @Override
    public String toString ()
    {
        return description;
    }
}
