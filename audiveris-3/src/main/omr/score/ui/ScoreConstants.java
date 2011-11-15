//----------------------------------------------------------------------------//
//                                                                            //
//                        S c o r e C o n s t a n t s                         //
//                                                                            //
//----------------------------------------------------------------------------//
// <editor-fold defaultstate="collapsed" desc="hdr">                          //
//  Copyright (C) Herve Bitteur 2000-2010. All rights reserved.               //
//  This software is released under the GNU General Public License.           //
//  Goto http://kenai.com/projects/audiveris to report bugs or suggestions.   //
//----------------------------------------------------------------------------//
// </editor-fold>
package omr.score.ui;

import omr.constant.Constant;
import omr.constant.ConstantSet;

import omr.ui.PixelCount;

/**
 * Class <code>ScoreConstants</code> gathers all constants related to the
 * display of a score.
 *
 * @author Hervé Bitteur
 */
public class ScoreConstants
{
    //~ Static fields/initializers ---------------------------------------------

    /** Specific application parameters */
    private static final Constants constants = new Constants();

    /** Height in pixels of a staff display */
    public static final int STAFF_AREA_HEIGHT = constants.staffAreaheight.getValue();

    /** Horizontal offset in pixels of the score origin */
    public static final int SCORE_INIT_X = constants.scoreInitX.getValue();

    /** Vertical offset in pixels of the score origin */
    public static final int SCORE_INIT_Y = constants.scoreInitY.getValue();

    /** Horizontal gutter in pixels between two systems */
    public static final int INTER_SYSTEM_WIDTH = constants.interSystemWidth.getValue();

    /** Vertical gutter in pixels between two systems */
    public static final int INTER_SYSTEM_HEIGHT = constants.interSystemHeight.getValue();

    /** Vertical distance in pixels between two lines of a standard staff :
       {@value} */
    public static final int INTER_LINE = 16;

    /** Horizontal gutter in pixels between two pages */
    public static final int INTER_PAGE = constants.interPage.getValue();

    /** Number of lines in a staff */
    public static final int LINE_NB = constants.lineNb.getValue();

    /** Height in pixels of one staff */
    public static final int STAFF_HEIGHT = (LINE_NB - 1) * INTER_LINE;

    /** Used to code fractions with an integer value, with a resolution of
       1/{@value} */
    public static final int BASE = 1024;

    //~ Constructors -----------------------------------------------------------

    //----------------//
    // ScoreConstants // Not meant to be instantiated
    //----------------//
    private ScoreConstants ()
    {
    }

    //~ Inner Classes ----------------------------------------------------------

    //-----------//
    // Constants //
    //-----------//
    private static final class Constants
        extends ConstantSet
    {
        //~ Instance fields ----------------------------------------------------

        Constant.Integer lineNb = new Constant.Integer(
            "Lines",
            5,
            "Number of lines in a staff");
        PixelCount       interPage = new PixelCount(
            200,
            "Horizontal gutter between two pages");
        PixelCount       interSystemWidth = new PixelCount(
            100,
            "Horizontal gutter between two systems");
        PixelCount       interSystemHeight = new PixelCount(
            200,
            "Vertical gutter between two systems");
        PixelCount       scoreInitX = new PixelCount(
            300,
            "Horizontal offset of the score origin");
        PixelCount       scoreInitY = new PixelCount(
            300,
            "Vertical offset of the score origin");
        PixelCount       staffAreaheight = new PixelCount(
            100,
            "Height of a staff display");
    }
}
