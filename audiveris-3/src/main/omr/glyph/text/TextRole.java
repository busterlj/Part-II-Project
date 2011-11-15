//----------------------------------------------------------------------------//
//                                                                            //
//                              T e x t R o l e                               //
//                                                                            //
//----------------------------------------------------------------------------//
// <editor-fold defaultstate="collapsed" desc="hdr">                          //
//  Copyright (C) Herve Bitteur 2000-2010. All rights reserved.               //
//  This software is released under the GNU General Public License.           //
//  Goto http://kenai.com/projects/audiveris to report bugs or suggestions.   //
//----------------------------------------------------------------------------//
// </editor-fold>
package omr.glyph.text;

import omr.constant.ConstantSet;

import omr.glyph.facets.Glyph;

import omr.log.Logger;

import omr.score.common.PageRectangle;
import omr.score.common.SystemPoint;
import omr.score.common.SystemRectangle;
import omr.score.entity.ScoreSystem;
import omr.score.entity.ScoreSystem.StaffPosition;
import omr.score.entity.Staff;
import omr.score.entity.Text;
import omr.score.entity.Text.CreatorText.CreatorType;

import omr.sheet.Scale;
import omr.sheet.Sheet;
import omr.sheet.SystemInfo;

/**
 * Class <code>TextRole</code> describes the role of a piece of text (typically
 * a sentence)
 *
 * @author Hervé Bitteur
 */
public enum TextRole {
    /** No role known */
    Unknown,
    /** (Part of) lyrics */
    Lyrics, 
    /** Title of the opus */
    Title, 
    /** Playing instruction */
    Direction, 
    /** Number for this opus */
    Number, 
    /** Name for the part */
    Name, 
    /** A creator (composer, etc...) */
    Creator, 
    /** Copyright notice */
    Rights;
    /**/

    /** Specific application parameters */
    private static final Constants constants = new Constants();

    /** Usual logger utility */
    private static final Logger logger = Logger.getLogger(TextRole.class);

    //-----------------//
    // getStringHolder //
    //-----------------//
    /**
     * A convenient method to forge a string to be used in lieu of text value
     * @param NbOfChars the number of characters desired
     * @return a dummy string of NbOfChars chars
     */
    public String getStringHolder (int NbOfChars)
    {
        if (NbOfChars > 1000) {
            logger.warning("Abnormal text length:" + NbOfChars);
            return "<<" + Integer.toString(NbOfChars) + ">>";
        } else {
            StringBuilder sb = new StringBuilder("[");

            while (sb.length() < (NbOfChars - 1)) {
                sb.append(toString())
                  .append("-");
            }

            return sb.substring(0, Math.max(NbOfChars - 1, 0)) + "]";
        }
    }

    //----------//
    // RoleInfo //
    //----------//
    public static class RoleInfo
    {
        final TextRole                     role;
        final Text.CreatorText.CreatorType creatorType;

        public RoleInfo (TextRole    role,
                         CreatorType creatorType)
        {
            this.role = role;
            this.creatorType = creatorType;
        }

        public RoleInfo (TextRole role)
        {
            this(role, null);
        }
    }

    //-----------//
    // guessRole //
    //-----------//
    /**
     * Try to infer the role of this textual item. For the time being, this is a
     * simple algorithm based on sentence location within the page,
     * but perhaps a neural network approach would better fit this task.
     * @param glyph the (only) sentence glyph
     * @param systemInfo the containing system
     * @return the role information inferred for the provided sentence glyph
     */
    static RoleInfo guessRole (Glyph      glyph,
                               SystemInfo systemInfo)
    {
        Sheet           sheet = systemInfo.getSheet();
        ScoreSystem     system = systemInfo.getScoreSystem();
        Scale           scale = system.getScale();
        PageRectangle   pageBox = scale.toUnits(glyph.getContourBox());
        SystemRectangle box = system.toSystemRectangle(pageBox);
        SystemPoint     left = new SystemPoint(box.x, box.y + (box.height / 2));
        SystemPoint     right = new SystemPoint(
            box.x + box.width,
            box.y + (box.height / 2));

        // First system in page?
        boolean       firstSystem = system.getId() == 1;

        // Last system in page?
        boolean       lastSystem = sheet.getSystems()
                                        .size() == system.getId();

        // Vertical position wrt staves
        StaffPosition staffPosition = system.getStaffPosition(left);

        // Vertical distance from staff
        Staff   staff = system.getStaffAt(left);
        int     staffDy = Math.abs(staff.getPageTopLeft().y - pageBox.y);
        boolean closeToStaff = staffDy <= scale.toUnits(constants.maxStaffDy);

        // Begins before the part?
        boolean leftOfStaves = system.isLeftOfStaves(left);

        // At the center of page width?
        int     maxCenterDx = scale.toUnits(constants.maxCenterDx);
        int     pageCenter = scale.pixelsToUnits(sheet.getWidth() / 2);
        boolean pageCentered = Math.abs(
            (pageBox.x + (pageBox.width / 2)) - pageCenter) <= maxCenterDx;

        // Right aligned with staves
        int     maxRightDx = scale.toUnits(constants.maxRightDx);
        boolean rightAligned = Math.abs(right.x - system.getDimension().width) <= maxRightDx;

        // Short Sentence?
        int     maxShortLength = scale.toUnits(constants.maxShortLength);
        boolean shortSentence = pageBox.width <= maxShortLength;

        int     minTitleHeight = scale.toUnits(constants.minTitleHeight);
        boolean highText = box.height >= minTitleHeight;

        if (logger.isFineEnabled()) {
            logger.fine(
                glyph + " firstSystem=" + firstSystem + " lastSystem=" +
                lastSystem + " staffPosition=" + staffPosition +
                " closeToStaff=" + closeToStaff + " leftOfStaves=" +
                leftOfStaves + " pageCentered=" + pageCentered +
                " rightAligned=" + rightAligned + " shortSentence=" +
                shortSentence + " highText=" + highText);
        }

        // Decisions ...
        switch (staffPosition) {
        case above : // Title, Number, Creator, Direction (Accord)

            if (leftOfStaves) {
                return new RoleInfo(
                    TextRole.Creator,
                    Text.CreatorText.CreatorType.lyricist);
            } else if (rightAligned) {
                return new RoleInfo(
                    TextRole.Creator,
                    Text.CreatorText.CreatorType.composer);
            } else if (closeToStaff) {
                return new RoleInfo(TextRole.Direction);
            } else if (pageCentered) { // Title, Number

                if (highText) {
                    return new RoleInfo(TextRole.Title);
                } else {
                    return new RoleInfo(TextRole.Number);
                }
            }

            break;

        case within : // Name, Lyrics, Direction

            if (leftOfStaves) {
                return new RoleInfo(TextRole.Name);
            } else if (shortSentence) {
                return new RoleInfo(TextRole.Direction);
            } else {
                return new RoleInfo(TextRole.Lyrics);
            }

        case below : // Copyright

            if (pageCentered && shortSentence && lastSystem) {
                return new RoleInfo(TextRole.Rights);
            }
        }

        return new RoleInfo(TextRole.Unknown);
    }

    //-----------//
    // Constants //
    //-----------//
    private static final class Constants
        extends ConstantSet
    {
        Scale.Fraction maxRightDx = new Scale.Fraction(
            2,
            "Maximum horizontal distance on the right end of the staff");
        Scale.Fraction maxCenterDx = new Scale.Fraction(
            30,
            "Maximum horizontal distance around center of page");
        Scale.Fraction maxShortLength = new Scale.Fraction(
            30,
            "Maximum length for a short sentence (no lyrics)");
        Scale.Fraction maxStaffDy = new Scale.Fraction(
            7,
            "Maximum distance above staff for a direction");
        Scale.Fraction minTitleHeight = new Scale.Fraction(
            3,
            "Minimum height for a title text");
    }
}
