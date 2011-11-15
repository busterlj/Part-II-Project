//----------------------------------------------------------------------------//
//                                                                            //
//                          S t a f f B u i l d e r                           //
//                                                                            //
//----------------------------------------------------------------------------//
// <editor-fold defaultstate="collapsed" desc="hdr">                          //
//  Copyright (C) Herve Bitteur 2000-2010. All rights reserved.               //
//  This software is released under the GNU General Public License.           //
//  Goto http://kenai.com/projects/audiveris to report bugs or suggestions.   //
//----------------------------------------------------------------------------//
// </editor-fold>
package omr.sheet;

import omr.glyph.GlyphLag;
import omr.glyph.GlyphSection;

import omr.log.Logger;

import omr.step.StepException;

import omr.stick.StickSection;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.ListIterator;

/**
 * Class <code>StaffBuilder</code> processes the (five) line areas, according to
 * the peaks found previously.
 *
 * @author Hervé Bitteur
 */
public class StaffBuilder
{
    //~ Static fields/initializers ---------------------------------------------

    /** Usual logger utility */
    private static final Logger logger = Logger.getLogger(StaffBuilder.class);

    //~ Instance fields --------------------------------------------------------

    /** Related sheet */
    private Sheet sheet;

    /** Related lag */
    private GlyphLag hLag;

    /** Vertex iterator */
    private ListIterator<GlyphSection> vi;

    /** To allow unique identifiers to staves (for debug only) */
    private int id;

    //~ Constructors -----------------------------------------------------------

    //--------------//
    // StaffBuilder //
    //--------------//
    /**
     * Create a staff retriever, based on an underlying horizontal lag..
     *
     * @param sheet the sheet we are analyzing
     * @param hLag  the horizontal lag
     * @param vi    the underlying vertex iterator
     */
    public StaffBuilder (Sheet                      sheet,
                         GlyphLag                   hLag,
                         ListIterator<GlyphSection> vi)
    {
        this.sheet = sheet;
        this.hLag = hLag;
        this.vi = vi;
    }

    //~ Methods ----------------------------------------------------------------

    /**
     * Create a staff info, using a list of related peaks that correspond to the
     * staff lines.
     *
     * @param peaks    the histogram peaks that belong to this staff area
     * @param interval the mean interval between peaks of this staff area
     * @return the created staff information
     * @throws StepException if anything goes wrong
     */
    public StaffInfo buildInfo (List<Peak> peaks,
                                double     interval)
        throws StepException
    {
        // Id for the newly created staff
        ++id;

        if (logger.isFineEnabled()) {
            logger.fine("Staff #" + id + " interval=" + interval);
        }

        // Specific staff scale
        Scale          scale = new Scale(
            (int) Math.rint(interval),
            sheet.getScale().mainFore());

        // Process each peak into a line of the set
        List<LineInfo> lines = new ArrayList<LineInfo>();

        for (Peak peak : peaks) {
            LineBuilder builder = new LineBuilder(
                hLag,
                peak.getTop(),
                peak.getBottom(),
                vi,
                sheet,
                scale);
            lines.add(builder.buildInfo());
        }

        // Retrieve left and right abscissa for the staff lines of the set We
        // use a kind of vote here, since one or two lines can be read as longer
        // than real, so we use the abscissa of the median.
        List<Integer> lefts = new ArrayList<Integer>();
        List<Integer> rights = new ArrayList<Integer>();

        for (LineInfo line : lines) {
            lefts.add(line.getLeft());
            rights.add(line.getRight());
        }

        Collections.sort(lefts);
        Collections.sort(rights);

        int left = lefts.get(2);
        int right = rights.get(2);

        if (logger.isFineEnabled()) {
            logger.fine("End of Staff #" + id + " " + this);
        }

        // Allocate the staff info
        return new StaffInfo(id, left, right, scale, sheet.getScale(), lines);
    }
}
