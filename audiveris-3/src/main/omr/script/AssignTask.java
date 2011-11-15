//----------------------------------------------------------------------------//
//                                                                            //
//                            A s s i g n T a s k                             //
//                                                                            //
//----------------------------------------------------------------------------//
// <editor-fold defaultstate="collapsed" desc="hdr">                          //
//  Copyright (C) Herve Bitteur 2000-2010. All rights reserved.               //
//  This software is released under the GNU General Public License.           //
//  Goto http://kenai.com/projects/audiveris to report bugs or suggestions.   //
//----------------------------------------------------------------------------//
// </editor-fold>
package omr.script;

import omr.glyph.Shape;
import omr.glyph.facets.Glyph;

import omr.lag.LagOrientation;

import omr.sheet.Sheet;

import omr.step.Step;

import java.util.Collection;

import javax.xml.bind.annotation.XmlAttribute;

/**
 * Class {@code AssignTask} assigns (or deassign) a shape to a collection of
 * glyphs.
 *
 * <p>Il the compound flag is set, a compound glyph may is composed from the
 * provided glyphs and assigned the shape. Otherwise, each provided glyph is
 * assigned the shape.</p>
 *
 * @author Hervé Bitteur
 */
public class AssignTask
    extends GlyphUpdateTask
{
    //~ Instance fields --------------------------------------------------------

    /** Assigned shape (or null for a deassignment) */
    @XmlAttribute
    private final Shape shape;

    /** True for a compound building */
    @XmlAttribute
    private final boolean compound;

    //~ Constructors -----------------------------------------------------------

    //------------//
    // AssignTask //
    //------------//
    /**
     * Create an assignment task
     *
     * @param shape the assigned shape (or null for a de-assignment)
     * @param compound true if all glyphs are to be merged into one compound
     * which is assigned to the given shape, false if each and every glyph is to
     * be assigned to the given shape
     * @param glyphs the collection of concerned glyphs
     * @param orientation orientation of the containing lag
     */
    public AssignTask (Shape             shape,
                       boolean           compound,
                       Collection<Glyph> glyphs,
                       LagOrientation    orientation)
    {
        super(orientation, glyphs);
        this.shape = shape;
        this.compound = compound;
    }

    //------------//
    // AssignTask //
    //------------//
    /**
     * Create an assignment task, for VERTICAL glyphs by default
     *
     * @param shape the assigned shape (or null for a de-assignment)
     * @param compound true if all glyphs are to be merged into one compound
     * which is assigned to the given shape, false if each and every glyph is to
     * be assigned to the given shape
     * @param glyphs the collection of concerned glyphs
     */
    public AssignTask (Shape             shape,
                       boolean           compound,
                       Collection<Glyph> glyphs)
    {
        this(shape, compound, glyphs, LagOrientation.VERTICAL);
    }

    //------------//
    // AssignTask //
    //------------//
    /**
     * Convenient way to create an deassignment task
     *
     * @param glyphs the collection of glyphs to deassign
     * @param orientation orientation of the containing lag
     */
    public AssignTask (Collection<Glyph> glyphs,
                       LagOrientation    orientation)
    {
        this(null, false, glyphs, orientation);
    }

    //------------//
    // AssignTask //
    //------------//
    /**
     * Convenient way to create an deassignment task for VERTICAL glyphs
     *
     * @param glyphs the collection of glyphs to deassign
     */
    public AssignTask (Collection<Glyph> glyphs)
    {
        this(null, false, glyphs, LagOrientation.VERTICAL);
    }

    //------------//
    // AssignTask //
    //------------//
    /** No-arg constructor needed for JAXB */
    protected AssignTask ()
    {
        shape = null;
        compound = false;
    }

    //~ Methods ----------------------------------------------------------------

    //------------------//
    // getAssignedShape //
    //------------------//
    /**
     * Report the assigned shape (for an assignment impact)
     * @return the assignedShape (null for a deasssignment)
     */
    public Shape getAssignedShape ()
    {
        return shape;
    }

    //------------//
    // isCompound //
    //------------//
    /**
     * Report whether the assignment is a compound
     * @return true for a compound assignment, false otherwise
     */
    public boolean isCompound ()
    {
        return compound;
    }

    //------//
    // core //
    //------//
    /**
     * {@inheritDoc}
     */
    @Override
    public void core (Sheet sheet)
        throws Exception
    {
        switch (orientation) {
        case HORIZONTAL :
            sheet.getHorizontalsBuilder()
                 .getController()
                 .syncAssign(this);

            break;

        case VERTICAL :
            sheet.getSymbolsController()
                 .syncAssign(this);
        }
    }

    //--------//
    // epilog //
    //--------//
    /**
     * {@inheritDoc}
     */
    @Override
    public void epilog (Sheet sheet)
    {
        switch (orientation) {
        case HORIZONTAL :
            sheet.getSheetSteps()
                 .rebuildFrom(Step.SYSTEMS, null, false);

            break;

        case VERTICAL :
            // We rebuild from VERTICALS is case of deassignment
            // And just from PATTERNS in case of assignment
            sheet.getSheetSteps()
                 .rebuildFrom(
                (shape == null) ? Step.VERTICALS : Step.PATTERNS,
                getImpactedSystems(sheet),
                false);
        }
    }

    //-----------------//
    // internalsString //
    //-----------------//
    @Override
    protected String internalsString ()
    {
        StringBuilder sb = new StringBuilder();
        sb.append(" assign");

        if (compound) {
            sb.append(" compound");
        }

        sb.append(" ")
          .append(orientation);

        if (shape != null) {
            sb.append(" ")
              .append(shape);
        } else {
            sb.append(" no-shape");
        }

        return sb + super.internalsString();
    }
}
