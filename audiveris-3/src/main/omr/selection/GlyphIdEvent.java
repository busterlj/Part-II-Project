//----------------------------------------------------------------------------//
//                                                                            //
//                          G l y p h I d E v e n t                           //
//                                                                            //
//----------------------------------------------------------------------------//
// <editor-fold defaultstate="collapsed" desc="hdr">                          //
//  Copyright (C) Herve Bitteur 2000-2010. All rights reserved.               //
//  This software is released under the GNU General Public License.           //
//  Goto http://kenai.com/projects/audiveris to report bugs or suggestions.   //
//----------------------------------------------------------------------------//
// </editor-fold>
package omr.selection;


/**
 * Class <code>GlyphIdEvent</code> represents a Glyph Id selection
 *
 * <dl>
 * <dt><b>Publishers:</b><dd>GlyphBoard, ShapeFocusBoard
 * <dt><b>Subscribers:</b><dd>GlyphLag, GlyphLagView
 * <dt><b>Readers:</b><dd>
 * </dl>
 *
 * @author Hervé Bitteur
 */
public class GlyphIdEvent
    extends GlyphLagEvent
{
    //~ Instance fields --------------------------------------------------------

    /** The selected glyph id, which may be null */
    public final Integer id;

    //~ Constructors -----------------------------------------------------------

    /**
     * Creates a new GlyphIdEvent object.
     *
     * @param source the entity that created this event
     * @param hint hint about event origin (or null)
     * @param movement the precise mouse movement
     * @param id the glyph id
     */
    public GlyphIdEvent (Object        source,
                         SelectionHint hint,
                         MouseMovement movement,
                         Integer       id)
    {
        super(source, hint, null);
        this.id = id;
    }

    //~ Methods ----------------------------------------------------------------

    //-----------//
    // getEntity //
    //-----------//
    @Override
    public Integer getData ()
    {
        return id;
    }
}
