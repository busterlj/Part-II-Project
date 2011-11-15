//----------------------------------------------------------------------------//
//                                                                            //
//                           S c r e e n P o i n t                            //
//                                                                            //
//----------------------------------------------------------------------------//
// <editor-fold defaultstate="collapsed" desc="hdr">                          //
//  Copyright (C) Herve Bitteur 2000-2010. All rights reserved.               //
//  This software is released under the GNU General Public License.           //
//  Goto http://kenai.com/projects/audiveris to report bugs or suggestions.   //
//----------------------------------------------------------------------------//
// </editor-fold>
package omr.ui.dnd;

import omr.score.common.SimplePoint;

import java.awt.Component;
import java.awt.Point;
import java.awt.Rectangle;

import javax.swing.SwingUtilities;

/**
 * Class {@code ScreenPoint} encapsulates a point defined with respect to the
 * screen
 *
 * <p> This specialization is used to take benefit of compiler checks, to
 * prevent the use of points with incorrect meaning or units. </p>
 *
 * @author Hervé Bitteur
 */
public class ScreenPoint
    extends SimplePoint
{
    //~ Constructors -----------------------------------------------------------

    //-------------//
    // ScreenPoint //
    //-------------//
    /**
     * Create a new ScreenPoint object
     */
    public ScreenPoint ()
    {
    }

    //-------------//
    // ScreenPoint //
    //-------------//
    /**
     * Creates a new ScreenPoint object, by cloning an untyped point
     *
     * @param x abscissa
     * @param y ordinate
     */
    public ScreenPoint (int x,
                        int y)
    {
        super(x, y);
    }

    //-------------//
    // ScreenPoint //
    //-------------//
    /**
     * Creates a new ScreenPoint object, using a local component-based point
     *
     * @param component the component to use as the base
     * @param point the component-based point
     */
    public ScreenPoint (Component component,
                        Point     point)
    {
        this(point.x, point.y);
        SwingUtilities.convertPointToScreen(this, component);
    }

    //~ Methods ----------------------------------------------------------------

    //---------------//
    // isInComponent //
    //---------------//
    /**
     * Check whether this screen point lies within the bound of the provided
     * component
     * @param component the provided component
     * @return true if within the component, false otherwise
     */
    public boolean isInComponent (Component component)
    {
        Rectangle bounds = new Rectangle(
            0,
            0,
            component.getWidth(),
            component.getHeight());

        return bounds.contains(getLocalPoint(component));
    }

    //---------------//
    // getLocalPoint //
    //---------------//
    /**
     * Get the corresponding local point wrt a containing component
     * @param component the provided component
     * @return the local point, wrt component top left corner
     */
    public Point getLocalPoint (Component component)
    {
        Point point = new Point(x, y);
        SwingUtilities.convertPointFromScreen(point, component);

        return point;
    }
}
