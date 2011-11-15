//----------------------------------------------------------------------------//
//                                                                            //
//                     G h o s t D r o p L i s t e n e r                      //
//                                                                            //
//----------------------------------------------------------------------------//
// <editor-fold defaultstate="collapsed" desc="hdr">                          //
//  Copyright (C) Herve Bitteur 2000-2010. All rights reserved.               //
//  This software is released under the GNU General Public License.           //
//  Goto http://kenai.com/projects/audiveris to report bugs or suggestions.   //
//----------------------------------------------------------------------------//
// </editor-fold>
package omr.ui.dnd;


/**
 * Interface {@code GhostDropListener} defines a listener interested in
 * {@link GhostDropEvent} instances
 *
 * @param <A> The type of action carried by the drop event
 * @author Hervé Bitteur (from Romain Guy's demo)
 */
public interface GhostDropListener<A>
{
    //~ Methods ----------------------------------------------------------------

    //--------------//
    // ghostDropped //
    //--------------//
    /**
     * Call-back function to received the drop event
     * @param e the handed event
     */
    public void ghostDropped (GhostDropEvent<A> e);
}
