//----------------------------------------------------------------------------//
//                                                                            //
//                             S t a f f I n f o                              //
//                                                                            //
//----------------------------------------------------------------------------//
// <editor-fold defaultstate="collapsed" desc="hdr">                          //
//  Copyright (C) Herve Bitteur 2000-2010. All rights reserved.               //
//  This software is released under the GNU General Public License.           //
//  Goto http://kenai.com/projects/audiveris to report bugs or suggestions.   //
//----------------------------------------------------------------------------//
// </editor-fold>
package omr.sheet;

import omr.log.Logger;

import omr.score.common.PixelPoint;

import java.awt.*;
import java.util.List;

/**
 * Class <code>StaffInfo</code> handles the physical details of a staff with its
 * lines.
 *
 * @author Hervé Bitteur
 */
public class StaffInfo
{
    //~ Static fields/initializers ---------------------------------------------

    /** Usual logger utility */
    private static final Logger logger = Logger.getLogger(StaffInfo.class);

    //~ Instance fields --------------------------------------------------------

    /** Lines of the set */
    private List<LineInfo> lines;

    /** Scale specific to this staff, since staves in a page may exhibit
       different scales. */
    private Scale specificScale;

    /** Global sheet scale, needed for many computations */
    private Scale scale;

    /** Bottom of staff related area */
    private int areaBottom = -1;

    /** Top of staff related area */
    private int areaTop = -1;

    /** For debug only */
    private int id;

    /** Left extrema */
    private int left;

    /** Right extrema */
    private int right;

    //~ Constructors -----------------------------------------------------------

    //-----------//
    // StaffInfo //
    //-----------//
    /**
     * Create info about a staff, with its contained staff lines
     *
     * @param id the id of the staff
     * @param left abscissa of the left side
     * @param right abscissa of the right side
     * @param specificScale specific scale detected for this staff
     * @param scale global sheet scale
     * @param lines the collection of contained staff lines
     */
    public StaffInfo (int            id,
                      int            left,
                      int            right,
                      Scale          specificScale,
                      Scale          scale,
                      List<LineInfo> lines)
    {
        this.id = id;
        this.left = left;
        this.right = right;
        this.specificScale = specificScale;
        this.scale = scale;
        this.lines = lines;
    }

    //~ Methods ----------------------------------------------------------------

    //---------------//
    // setAreaBottom //
    //---------------//
    /**
     * Define the ordinate of the bottom of the staff area
     *
     * @param val bottom ordinate
     */
    public void setAreaBottom (int val)
    {
        areaBottom = val;
    }

    //---------------//
    // getAreaBottom //
    //---------------//
    /**
     * Selector for bottom of area
     *
     * @return area bottom
     */
    public int getAreaBottom ()
    {
        return areaBottom;
    }

    //------------//
    // setAreaTop //
    //------------//
    /**
     * Define the ordinate for top of staff area
     *
     * @param val top ordinate
     */
    public void setAreaTop (int val)
    {
        areaTop = val;
    }

    //------------//
    // getAreaTop //
    //------------//
    /**
     * Selector for area top ordinate
     *
     * @return area top ordinate
     */
    public int getAreaTop ()
    {
        return areaTop;
    }

    //--------------//
    // getFirstLine //
    //--------------//
    /**
     * Report the first line in the series
     *
     * @return the first line
     */
    public LineInfo getFirstLine ()
    {
        return lines.get(0);
    }

    //-----------//
    // getHeight //
    //-----------//
    /**
     * Report the mean height of the staff, between first and last line
     *
     * @return the mean staff height
     */
    public int getHeight ()
    {
        return getSpecificScale()
                   .interline() * 4;
    }

    //-------------//
    // getLastLine //
    //-------------//
    /**
     * Report the last line in the series
     *
     * @return the last line
     */
    public LineInfo getLastLine ()
    {
        return lines.get(lines.size() - 1);
    }

    //---------//
    // getLeft //
    //---------//
    /**
     * Report the abscissa of the left side of the staff
     *
     * @return the left abscissa
     */
    public int getLeft ()
    {
        return left;
    }

    //----------//
    // getLines //
    //----------//
    /**
     * Report the list of line areas
     *
     * @return the list of lines in this staff
     */
    public List<LineInfo> getLines ()
    {
        return lines;
    }

    //----------//
    // getRight //
    //----------//
    /**
     * Report the abscissa of the right side of the staff
     *
     * @return the right abscissa
     */
    public int getRight ()
    {
        return right;
    }

    //----------//
    // getScale //
    //---------//
    /**
     * Report the global sheet scale.
     *
     * @return the sheet global scale
     */
    public Scale getScale ()
    {
        return scale;
    }

    //------------------//
    // getSpecificScale //
    //------------------//
    /**
     * Report the <b>specific</b> staff scale, which may have a different
     * interline value than the page average.
     *
     * @return the staff scale
     */
    public Scale getSpecificScale ()
    {
        if (specificScale != null) {
            // Return the specific scale of this staff
            return specificScale;
        } else {
            // Return the scale of the sheet
            logger.warning("No specific scale available");

            return null;
        }
    }

    //---------//
    // cleanup //
    //---------//
    /**
     * Forward the cleaning order to each of the staff lines
     */
    public void cleanup ()
    {
        for (LineInfo line : lines) {
            line.cleanup();
        }
    }

    //------//
    // dump //
    //------//
    /**
     * A utility meant for debugging
     */
    public void dump ()
    {
        System.out.println(
            "StaffInfo" + id + " left=" + left + " right=" + right);

        int i = 0;

        for (LineInfo line : lines) {
            System.out.println(" LineInfo" + i++ + " " + line.toString());
        }
    }

    //-----------------//
    // pitchPositionOf //
    //-----------------//
    /**
     * Compute the pitch position of a pixel point
     *
     * @param pt the pixel point
     * @return the pitch position
     */
    public double pitchPositionOf (PixelPoint pt)
    {
        int top = getFirstLine()
                      .yAt(pt.x);
        int bottom = getLastLine()
                         .yAt(pt.x);

        return (4.0d * ((2 * pt.y) - bottom - top)) / (bottom - top);
    }

    //--------//
    // render //
    //--------//
    /**
     * Paint the staff lines.
     *
     * @param g the graphics context
     * @return true if something has been drawn
     */
    public boolean render (Graphics g)
    {
        LineInfo firstLine = getFirstLine();
        LineInfo lastLine = getLastLine();

        if ((firstLine != null) && (lastLine != null)) {
            // Check that top of staff is visible
            final int yTopLeft = firstLine.yAt(left);
            final int yTopRight = firstLine.yAt(right);

            // Check with the clipping region
            Rectangle clip = g.getClipBounds();

            if ((clip.y + clip.height) < Math.max(yTopLeft, yTopRight)) {
                return false;
            }

            // Check that bottom of staff is visible
            final int yBottomLeft = lastLine.yAt(left);
            final int yBottomRight = lastLine.yAt(right);

            if (clip.y > Math.max(yBottomLeft, yBottomRight)) {
                return false;
            }

            // Paint each horizontal line in the set
            for (LineInfo line : lines) {
                line.render(g, left, right);
            }

            // Left vertical line
            g.drawLine(left, yTopLeft, left, yBottomLeft);

            // Right vertical line
            g.drawLine(right, yTopRight, right, yBottomRight);

            return true;
        } else {
            return false;
        }
    }

    //----------//
    // toString //
    //----------//
    /**
     * Report a readable description
     *
     * @return a string based on main parameters
     */
    @Override
    public String toString ()
    {
        StringBuilder sb = new StringBuilder();
        sb.append("{StaffInfo")
          .append(" id=")
          .append(id)
          .append(" left=")
          .append(left)
          .append(" right=")
          .append(right);

        if (specificScale != null) {
            sb.append(" specificScale=")
              .append(specificScale.interline());
        }

        sb.append("}");

        return sb.toString();
    }
}
