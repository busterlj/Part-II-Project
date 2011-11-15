//----------------------------------------------------------------------------//
//                                                                            //
//                             A u d i v e r i s                              //
//                                                                            //
//----------------------------------------------------------------------------//
// <editor-fold defaultstate="collapsed" desc="hdr">                          //
//  Copyright (C) Herve Bitteur 2000-2010. All rights reserved.               //
//  This software is released under the GNU General Public License.           //
//  Goto http://kenai.com/projects/audiveris to report bugs or suggestions.   //
//----------------------------------------------------------------------------//
// </editor-fold>
import omr.WellKnowns;

/**
 * Class <code>Audiveris</code> is simply the entry point to OMR, which
 * delegates the call to {@link omr.Main#main}.
 *
 * @author Hervé Bitteur
 */
public final class Audiveris
{
    //~ Constructors -----------------------------------------------------------

    //-----------//
    // Audiveris //
    //-----------//
    /** To avoid instantiation */
    private Audiveris ()
    {
    }

    //~ Methods ----------------------------------------------------------------

    //------//
    // main //
    //------//
    /**
     * The main entry point, which just calls {@link omr.Main#main}
     *
     * @param args These args are simply passed to Main
     */
    public static void main (final String[] args)
    {
        // We need class WellKnowns to be elaborated before class Main
        WellKnowns.ensureLoaded();

        // Then we call Main...
        omr.Main.main(args);
    }
}
