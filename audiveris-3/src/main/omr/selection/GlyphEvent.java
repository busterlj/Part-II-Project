//----------------------------------------------------------------------------//
//                                                                            //
//                            G l y p h E v e n t                             //
//                                                                            //
//----------------------------------------------------------------------------//
// <editor-fold defaultstate="collapsed" desc="hdr">                          //
//  Copyright (C) Herve Bitteur 2000-2010. All rights reserved.               //
//  This software is released under the GNU General Public License.           //
//  Goto http://kenai.com/projects/audiveris to report bugs or suggestions.   //
//----------------------------------------------------------------------------//
// </editor-fold>
package omr.selection;

import omr.glyph.facets.Glyph;

/**
 * Class <code>GlyphEvent</code> represents a Glyph selection
 *
 * <dl>
 * <dt><b>Publishers:</b><dd>ErrorsEditor, GlyphBoard, GlyphBrowser, GlyphLag,
 * GlyphLagView, GlyphMenu, GlyphModel, SymbolsEditor
 * <dt><b>Subscribers:</b><dd>EvaluationBoard, GlyphBoard, GlyphBrowser,
 * GlyphLag, GlyphLagView, HorizontalsBuilder CheckBoards, ShapeFocusBoard,
 * SymbolGlyphBoard, SystemsBuilder, VerticalsBuilder CheckBoard
 * <dt><b>Readers:</b><dd>EvaluationBoard, GlyphBoard, GlyphMenu, TextAreaBrowser
 * </dl>
 *
 * @author Hervé Bitteur
 */
public class GlyphEvent
    extends GlyphLagEvent
{
    //~ Instance fields --------------------------------------------------------

    /** The selected glyph, which may be null */
    public final Glyph glyph;

    //~ Constructors -----------------------------------------------------------

    //------------//
    // GlyphEvent //
    //------------//
    /**
     * Creates a new GlyphEvent object.
     *
     * @param source the entity that created this event
     * @param hint hint about event origin (or null)
     * @param movement the mouse movement
     * @param glyph the selected glyph (or null)
     */
    public GlyphEvent (Object        source,
                       SelectionHint hint,
                       MouseMovement movement,
                       Glyph         glyph)
    {
        super(source, hint, movement);
        this.glyph = glyph;
    }

    //~ Methods ----------------------------------------------------------------

    //-----------//
    // getEntity //
    //-----------//
    @Override
    public Glyph getData ()
    {
        return glyph;
    }
}
