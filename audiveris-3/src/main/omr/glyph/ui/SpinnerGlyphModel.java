//----------------------------------------------------------------------------//
//                                                                            //
//                     S p i n n e r G l y p h M o d e l                      //
//                                                                            //
//----------------------------------------------------------------------------//
// <editor-fold defaultstate="collapsed" desc="hdr">                          //
//  Copyright (C) Herve Bitteur 2000-2010. All rights reserved.               //
//  This software is released under the GNU General Public License.           //
//  Goto http://kenai.com/projects/audiveris to report bugs or suggestions.   //
//----------------------------------------------------------------------------//
// </editor-fold>
package omr.glyph.ui;

import omr.glyph.GlyphLag;
import omr.glyph.facets.Glyph;

import omr.log.Logger;
import static omr.ui.field.SpinnerUtilities.*;

import omr.util.Implement;
import omr.util.Predicate;

import java.util.*;

import javax.swing.*;

/**
 * Class <code>SpinnerGlyphModel</code> is a spinner model backed by a {@link
 * GlyphLag} and a potential additional glyph collection (that belongs to a
 * {@link GlyphLagView}. Any modification in the lag is thus transparently
 * handled, since the lag <b>is</b> the model. <p>A glyph {@link Predicate} can
 * be assigned to this SpinnerGlyphModel at construction time in order to
 * restrict the population of glyphs in the spinner. This class is used by
 * {@link GlyphBoard} only, but is not coupled with it.
 *
 * @author Hervé Bitteur
 */
public class SpinnerGlyphModel
    extends AbstractSpinnerModel
{
    //~ Static fields/initializers ---------------------------------------------

    /** Usual logger utility */
    private static final Logger logger = Logger.getLogger(
        SpinnerGlyphModel.class);

    //~ Instance fields --------------------------------------------------------

    /** Underlying glyph lag */
    private final GlyphLag lag;

    /** Additional glyph collection is any */
    private final Collection<?extends Glyph> specificGlyphs;

    /** Additional predicate if any */
    private final Predicate<Glyph> predicate;

    /** Current glyph id */
    private Integer currentId;

    //~ Constructors -----------------------------------------------------------

    //-------------------//
    // SpinnerGlyphModel //
    //-------------------//
    /**
     * Creates a new SpinnerGlyphModel object, on all lag glyphs
     *
     * @param lag the underlying glyph lag
     */
    public SpinnerGlyphModel (GlyphLag lag)
    {
        this(lag, null, null);
    }

    //-------------------//
    // SpinnerGlyphModel //
    //-------------------//
    /**
     * Creates a new SpinnerGlyphModel object, with a related glyph predicate
     *
     * @param lag the underlying glyph lag
     * @param specificGlyphs additional glyph collection, or null
     * @param predicate predicate of glyph, or null
     */
    public SpinnerGlyphModel (GlyphLag                   lag,
                              Collection<?extends Glyph> specificGlyphs,
                              Predicate<Glyph>           predicate)
    {
        if (lag == null) {
            throw new IllegalArgumentException(
                "SpinnerGlyphModel expects non-null glyph lag");
        }

        this.lag = lag;
        this.predicate = predicate;

        if (specificGlyphs != null) {
            this.specificGlyphs = specificGlyphs;
        } else {
            this.specificGlyphs = new ArrayList<Glyph>(0);
        }

        currentId = NO_VALUE;
    }

    //~ Methods ----------------------------------------------------------------

    //--------------//
    // getNextValue //
    //--------------//
    /**
     * Return the next legal glyph id in the sequence that comes after the glyph
     * id returned by <code>getValue()</code>. If the end of the sequence has
     * been reached then return null.
     *
     * @return the next legal glyph id or null if one doesn't exist
     */
    @Implement(SpinnerModel.class)
    public Object getNextValue ()
    {
        final int cur = currentId.intValue();

        if (logger.isFineEnabled()) {
            logger.fine("getNextValue cur=" + cur);
        }

        if (cur == NO_VALUE) {
            // Return first suitable glyph in lag
            for (Glyph glyph : lag.getAllGlyphs()) {
                if ((predicate == null) || predicate.check(glyph)) {
                    return glyph.getId();
                }
            }

            // Just in case, fall back to specifics
            for (Glyph glyph : specificGlyphs) {
                if ((predicate == null) || predicate.check(glyph)) {
                    return glyph.getId();
                }
            }

            return null;
        } else {
            // Return first suitable glyph after current glyph in lag
            boolean found = false;

            for (Glyph glyph : lag.getAllGlyphs()) {
                if (!found) {
                    if (glyph.getId() == cur) {
                        found = true;
                    }
                } else if ((predicate == null) || predicate.check(glyph)) {
                    return glyph.getId();
                }
            }

            // Fall back to specifics
            for (Glyph glyph : specificGlyphs) {
                if (!found) {
                    if (glyph.getId() == cur) {
                        found = true;
                    }
                } else if ((predicate == null) || predicate.check(glyph)) {
                    return glyph.getId();
                }
            }

            return null;
        }
    }

    //------------------//
    // getPreviousValue //
    //------------------//
    /**
     * Return the legal glyph id in the sequence that comes before the glyph id
     * returned by <code>getValue()</code>.  If the end of the sequence has been
     * reached then return null.
     *
     * @return the previous legal value or null if one doesn't exist
     */
    @Implement(SpinnerModel.class)
    public Object getPreviousValue ()
    {
        Glyph     prevGlyph = null;
        final int cur = currentId.intValue();

        if (logger.isFineEnabled()) {
            logger.fine("getPreviousValue cur=" + cur);
        }

        if (cur == NO_VALUE) {
            return NO_VALUE;
        }

        // Lag
        for (Glyph glyph : lag.getAllGlyphs()) {
            if (glyph.getId() == cur) {
                return (prevGlyph != null) ? prevGlyph.getId() : NO_VALUE;
            }

            // Should we remember this as (suitable) previous glyph ?
            if ((predicate == null) || predicate.check(glyph)) {
                prevGlyph = glyph;
            }
        }

        // Specifics
        for (Glyph glyph : specificGlyphs) {
            if (glyph.getId() == cur) {
                return (prevGlyph != null) ? prevGlyph.getId() : NO_VALUE;
            }

            // Should we remember this as (suitable) previous glyph ?
            if ((predicate == null) || predicate.check(glyph)) {
                prevGlyph = glyph;
            }
        }

        return null;
    }

    //----------//
    // setValue //
    //----------//
    /**
     * Changes current glyph id of the model.  If the glyph id is illegal then
     * an <code>IllegalArgumentException</code> is thrown.
     *
     * @param value the value to set
     * @exception IllegalArgumentException if <code>value</code> isn't allowed
     */
    @Implement(SpinnerModel.class)
    public void setValue (Object value)
    {
        if (logger.isFineEnabled()) {
            logger.fine("setValue value=" + value);
        }

        Integer id = (Integer) value;
        boolean ok;

        if (id == NO_VALUE) {
            ok = true;
        } else {
            // Lag
            Glyph glyph = lag.getGlyph(id);

            if (glyph != null) {
                if (predicate != null) {
                    ok = predicate.check(glyph);
                } else {
                    ok = true;
                }
            } else {
                // Specifics
                int intId = id.intValue();
                ok = false;

                for (Glyph g : specificGlyphs) {
                    if (g.getId() == intId) {
                        if (predicate != null) {
                            ok = predicate.check(g);
                        } else {
                            ok = true;
                        }

                        break;
                    }
                }
            }
        }

        if (ok) {
            currentId = id;
            fireStateChanged();
        } else {
            logger.warning("Invalid spinner element : " + id);
        }
    }

    //----------//
    // getValue //
    //----------//
    /**
     * The <i>current element</i> of the sequence.
     *
     * @return the current spinner value.
     */
    @Implement(SpinnerModel.class)
    public Object getValue ()
    {
        if (logger.isFineEnabled()) {
            logger.fine("getValue currentId=" + currentId);
        }

        return currentId;
    }
}
