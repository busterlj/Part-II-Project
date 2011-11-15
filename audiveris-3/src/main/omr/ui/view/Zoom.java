//----------------------------------------------------------------------------//
//                                                                            //
//                                  Z o o m                                   //
//                                                                            //
//----------------------------------------------------------------------------//
// <editor-fold defaultstate="collapsed" desc="hdr">                          //
//  Copyright (C) Herve Bitteur 2000-2010. All rights reserved.               //
//  This software is released under the GNU General Public License.           //
//  Goto http://kenai.com/projects/audiveris to report bugs or suggestions.   //
//----------------------------------------------------------------------------//
// </editor-fold>
package omr.ui.view;

import omr.constant.Constant;
import omr.constant.ConstantSet;

import omr.log.Logger;

import java.awt.Dimension;
import java.awt.Point;
import java.awt.Rectangle;
import java.util.HashSet;
import java.util.Set;

import javax.swing.event.*;

/**
 * Class <code>Zoom</code> encapsulates a zoom ratio, which is typically
 * the ratio between display values (such as the size of the display of an
 * entity) and model values (such as the size of the entity itself). For
 * example, a Zoom with ratio set to a 2.0 value would double the display
 * of a given entity.
 *
 * <p>Since this class is meant to be used when handling display tasks, it
 * also provides utility methods to go back and forth between display and
 * model values, for simple items such as primitive data (double), Point,
 * Dimension and Rectangle.
 *
 * <p>It handles an internal collection of change listeners, that are
 * entities registered to be notified whenever a new ratio value is set.
 *
 * <p>A new value is programmatically set by calling the {@link #setRatio}
 * method. The newly defined ratio value is then pushed to all registered
 * observers.
 *
 * <p>A {@link LogSlider} can be connected to this zoom entity, to provide
 * UI for both output and input.
 *
 * @author Hervé Bitteur
 */
public class Zoom
{
    //~ Static fields/initializers ---------------------------------------------

    /** Specific application parameters */
    private static final Constants constants = new Constants();

    /** Usual logger utility */
    private static final Logger logger = Logger.getLogger(Zoom.class);

    // To assign a unique Id
    private static int            globalId;

    //~ Instance fields --------------------------------------------------------

    /** Unique event, created lazily */
    protected ChangeEvent changeEvent = null;

    /** Potential logarithmic slider to drive this zoom */
    protected LogSlider slider;

    /** Collection of event listeners */
    protected Set<ChangeListener> listeners = new HashSet<ChangeListener>();

    /** Current ratio value */
    protected double ratio;

    // Unique Id (to ease debugging)
    private int id = ++globalId;

    //~ Constructors -----------------------------------------------------------

    //------//
    // Zoom //
    //------//
    /**
     * Create a zoom entity, with a default ratio value of 1.
     */
    public Zoom ()
    {
        this(1);
    }

    //------//
    // Zoom //
    //------//
    /**
     * Create a zoom entity, with the provided initial ratio value.
     *
     * @param ratio the initial ratio value
     */
    public Zoom (double ratio)
    {
        if (logger.isFineEnabled()) {
            logger.fine("Zoom created with ratio " + ratio);
        }

        setRatio(ratio);
    }

    //------//
    // Zoom //
    //------//
    /**
     * Create a zoom entity, with the provided initial ratio value. and a
     * related slider
     *
     * @param slider the related slider
     * @param ratio the initial ratio value
     */
    public Zoom (LogSlider slider,
                 double    ratio)
    {
        if (logger.isFineEnabled()) {
            logger.fine(
                "Zoom created" + " slider=" + slider + " ratio=" + ratio);
        }

        setSlider(slider);
        setRatio(ratio);
    }

    //~ Methods ----------------------------------------------------------------

    //----------//
    // setRatio //
    //----------//
    /**
     * Change the display zoom ratio. Nota, if the zoom is coupled with a
     * slider, this slider has the final word concerning the precise zoom
     * value, since the slider uses integer (or fractional) values.
     *
     * @param ratio the new ratio
     */
    public void setRatio (double ratio)
    {
        if (logger.isFineEnabled()) {
            logger.fine("setRatio ratio=" + ratio);
        }

        // Propagate to slider (useful to keep slider in sync when ratio is
        // set programmatically)
        if (slider != null) {
            slider.setDoubleValue(ratio);
        } else {
            forceRatio(ratio);
        }
    }

    //----------//
    // getRatio //
    //----------//
    /**
     * Return the current zoom ratio. Ratio is defined as display / source.
     *
     * @return the ratio
     */
    public double getRatio ()
    {
        return ratio;
    }

    //-----------//
    // setSlider //
    //-----------//
    /**
     * Define a related logarithmic slider, as a UI to adjust the zoom
     * value
     *
     * @param slider the related slider UI
     */
    public void setSlider (final LogSlider slider)
    {
        this.slider = slider;

        if (logger.isFineEnabled()) {
            logger.fine("setSlider");
        }

        if (slider != null) {
            slider.setDoubleValue(ratio);

            slider.addChangeListener(
                new ChangeListener() {
                        public void stateChanged (ChangeEvent e)
                        {
                            // Forward the new zoom ratio
                            if (constants.continuousSliderReading.getValue() ||
                                !slider.getValueIsAdjusting()) {
                                double newRatio = slider.getDoubleValue();

                                if (logger.isFineEnabled()) {
                                    logger.fine(
                                        "Slider firing zoom newRatio=" +
                                        newRatio);
                                }

                                // Stop condition to avoid endless loop between
                                // slider and zoom
                                if (Math.abs(newRatio - ratio) > .001) {
                                    forceRatio(newRatio);
                                }
                            }
                        }
                    });
        }
    }

    //-------------------//
    // addChangeListener //
    //-------------------//
    /**
     * Register a change listener, to be notified when the zoom value is
     * changed
     *
     * @param listener the listener to be notified
     */
    public void addChangeListener (ChangeListener listener)
    {
        listeners.add(listener);

        if (logger.isFineEnabled()) {
            logger.fine(
                "addChangeListener " + listener + " -> " + listeners.size());
        }
    }

    //------------------//
    // fireStateChanged //
    //------------------//
    /**
     * In charge of forwarding the change notification to all registered
     * listeners
     */
    public void fireStateChanged ()
    {
        for (ChangeListener listener : listeners) {
            if (changeEvent == null) {
                changeEvent = new ChangeEvent(this);
            }

            if (logger.isFineEnabled()) {
                logger.fine(this + " Firing " + listener);
            }

            listener.stateChanged(changeEvent);
        }
    }

    //----------------------//
    // removeChangeListener //
    //----------------------//
    /**
     * Unregister a change listener
     *
     * @param listener the listener to remove
     * @return true if actually removed
     */
    public boolean removeChangeListener (ChangeListener listener)
    {
        return listeners.remove(listener);
    }

    //-------//
    // scale //
    //-------//
    /**
     * Scale all the elements of provided point
     *
     * @param pt the point to be scaled
     */
    public void scale (Point pt)
    {
        pt.x = scaled(pt.x);
        pt.y = scaled(pt.y);
    }

    //-------//
    // scale //
    //-------//
    /**
     * Scale all the elements of provided dimension
     *
     * @param dim the dimension to be scaled
     */
    public void scale (Dimension dim)
    {
        dim.width = scaled(dim.width);
        dim.height = scaled(dim.height);
    }

    //-------//
    // scale //
    //-------//
    /**
     * Scale all the elements of provided rectangle
     *
     * @param rect the rectangle to be scaled
     */
    public void scale (Rectangle rect)
    {
        rect.x = scaled(rect.x);
        rect.y = scaled(rect.y);
        rect.width = scaled(rect.width);
        rect.height = scaled(rect.height);
    }

    //--------//
    // scaled //
    //--------//
    /**
     * Coordinate computation, Source -> Display
     *
     * @param val a source value
     *
     * @return the (scaled) display value
     */
    public int scaled (double val)
    {
        return (int) Math.rint(val * ratio);
    }

    //--------//
    // scaled //
    //--------//
    /**
     * Coordinate computation, Source -> Display
     *
     * @param dim source dimension
     *
     * @return the corresponding (scaled) dimension
     */
    public Dimension scaled (Dimension dim)
    {
        Dimension d = new Dimension(dim);
        scale(d);

        return d;
    }

    //--------//
    // scaled //
    //--------//
    /**
     * Coordinate computation, Source -> Display
     *
     * @param rect source rectangle
     *
     * @return the corresponding (scaled) rectangle
     */
    public Rectangle scaled (Rectangle rect)
    {
        Rectangle r = new Rectangle(rect);
        scale(r);

        return r;
    }

    //----------//
    // toString //
    //----------//
    /**
     * Report a quick description
     *
     * @return the zoom information
     */
    @Override
    public String toString ()
    {
        return "{Zoom#" + id + " listeners=" + listeners.size() + " ratio=" +
               ratio + "}";
    }

    //-------------//
    // truncScaled //
    //-------------//
    /**
     * Coordinate computation, Source -> Display, but with a truncation
     * rather than rounding.
     *
     * @param val a source value
     *
     * @return the (scaled) display value
     */
    public int truncScaled (double val)
    {
        return (int) (val * ratio);
    }

    //---------------//
    // truncUnscaled //
    //---------------//
    /**
     * Coordinate computation, Display -> Source, but with a truncation
     * rather than rounding.
     *
     * @param val a display value
     *
     * @return the corresponding (unscaled) source coordinate
     */
    public int truncUnscaled (double val)
    {
        return (int) (val / ratio);
    }

    //---------//
    // unscale //
    //---------//
    /**
     * Unscale a point
     * @param pt the point to unscale
     */
    public void unscale (Point pt)
    {
        pt.x = unscaled(pt.x);
        pt.y = unscaled(pt.y);
    }

    //----------//
    // unscaled //
    //----------//
    /**
     * Coordinate computation Display -> Source
     *
     * @param val a display value
     *
     * @return the corresponding (unscaled) source coordinate
     */
    public int unscaled (double val)
    {
        return (int) Math.rint(val / ratio);
    }

    //------------//
    // forceRatio //
    //------------//
    /**
     * Impose and propagate the new ratio
     *
     * @param ratio the new ratio
     */
    private void forceRatio (double ratio)
    {
        if (logger.isFineEnabled()) {
            logger.fine("forceRatio ratio=" + ratio);
        }

        this.ratio = ratio;

        // Propagate to listeners
        fireStateChanged();
    }

    //~ Inner Classes ----------------------------------------------------------

    //-----------//
    // Constants //
    //-----------//
    private static final class Constants
        extends ConstantSet
    {
        //~ Instance fields ----------------------------------------------------

        Constant.Boolean continuousSliderReading = new Constant.Boolean(
            true,
            "Should we allow continuous reading of the zoom slider");
    }
}
