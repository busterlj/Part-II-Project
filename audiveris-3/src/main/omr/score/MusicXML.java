//----------------------------------------------------------------------------//
//                                                                            //
//                              M u s i c X M L                               //
//                                                                            //
//----------------------------------------------------------------------------//
// <editor-fold defaultstate="collapsed" desc="hdr">                          //
//  Copyright (C) Herve Bitteur 2000-2010. All rights reserved.               //
//  This software is released under the GNU General Public License.           //
//  Goto http://kenai.com/projects/audiveris to report bugs or suggestions.   //
//----------------------------------------------------------------------------//
// </editor-fold>
package omr.score;

import omr.glyph.Shape;
import static omr.glyph.Shape.*;

import omr.log.Logger;

import omr.score.common.SystemPoint;
import omr.score.entity.LyricsItem;
import omr.score.entity.Note;
import omr.score.entity.Staff;

import proxymusic.*;

import java.lang.String; // Do not remove this line!
import java.math.BigDecimal;

import javax.xml.bind.JAXBElement;

/**
 * Class <code>MusicXML</code> gathers symbols related to the MusicXML data
 *
 * @author Hervé Bitteur
 */
public class MusicXML
{
    //~ Static fields/initializers ---------------------------------------------

    /** Usual logger utility */
    private static final Logger logger = Logger.getLogger(MusicXML.class);

    /** Names of the various note types used in MusicXML */
    private static final String[] noteTypeNames = new String[] {
                                                      "256th", "128th", "64th",
                                                      "32nd", "16th", "eighth",
                                                      "quarter", "half", "whole",
                                                      "breve", "long"
                                                  };

    //~ Constructors -----------------------------------------------------------

    //---------------//
    // ScoreExporter //
    //---------------//
    /**
     * Not meant to be instantiated
     */
    private MusicXML ()
    {
    }

    //~ Methods ----------------------------------------------------------------

    //-----------------------//
    // getArticulationObject //
    //-----------------------//
    public static JAXBElement<?> getArticulationObject (Shape shape)
    {
        //<!ELEMENT articulations
        //	((accent | strong-accent | staccato | tenuto |
        //	  detached-legato | staccatissimo | spiccato |
        //	  scoop | plop | doit | falloff | breath-mark | 
        //	  caesura | stress | unstress | other-articulation)*)>
        ObjectFactory  factory = new ObjectFactory();
        EmptyPlacement ep = factory.createEmptyPlacement();

        switch (shape) {
        case DOT :
        case STACCATO :
            return factory.createArticulationsStaccato(ep);

        case ACCENT :
            return factory.createArticulationsAccent(ep);

        case STRONG_ACCENT :

            // Type for strong accent: either up (^) or down (v)
            // For the time being we recognize only up ones
            StrongAccent strongAccent = factory.createStrongAccent();

            if (shape == Shape.STRONG_ACCENT) {
                strongAccent.setType(UpDown.UP);
            }

            return factory.createArticulationsStrongAccent(strongAccent);

        case TENUTO :
            return factory.createArticulationsTenuto(ep);

        case STACCATISSIMO :
            return factory.createArticulationsStaccatissimo(ep);

            /** TODO: implement related shapes
               case BREATH_MARK :
               case CAESURA :
             */
        }

        logger.severe("Unsupported ornament shape:" + shape);

        return null;
    }

    //-------------------//
    // getDynamicsObject //
    //-------------------//
    public static JAXBElement<?> getDynamicsObject (Shape shape)
    {
        ObjectFactory factory = new ObjectFactory();
        Empty         empty = factory.createEmpty();

        switch (shape) {
        case DYNAMICS_F :
            return factory.createDynamicsF(empty);

        case DYNAMICS_FF :
            return factory.createDynamicsFf(empty);

        case DYNAMICS_FFF :
            return factory.createDynamicsFff(empty);

        case DYNAMICS_FFFF :
            return factory.createDynamicsFfff(empty);

        case DYNAMICS_FFFFF :
            return factory.createDynamicsFffff(empty);

        case DYNAMICS_FFFFFF :
            return factory.createDynamicsFfffff(empty);

        case DYNAMICS_FP :
            return factory.createDynamicsFp(empty);

        case DYNAMICS_FZ :
            return factory.createDynamicsFz(empty);

        case DYNAMICS_MF :
            return factory.createDynamicsMf(empty);

        case DYNAMICS_MP :
            return factory.createDynamicsMp(empty);

        case DYNAMICS_P :
            return factory.createDynamicsP(empty);

        case DYNAMICS_PP :
            return factory.createDynamicsPp(empty);

        case DYNAMICS_PPP :
            return factory.createDynamicsPpp(empty);

        case DYNAMICS_PPPP :
            return factory.createDynamicsPppp(empty);

        case DYNAMICS_PPPPP :
            return factory.createDynamicsPpppp(empty);

        case DYNAMICS_PPPPPP :
            return factory.createDynamicsPppppp(empty);

        case DYNAMICS_RF :
            return factory.createDynamicsRf(empty);

        case DYNAMICS_RFZ :
            return factory.createDynamicsRfz(empty);

        case DYNAMICS_SF :
            return factory.createDynamicsSf(empty);

        case DYNAMICS_SFFZ :
            return factory.createDynamicsSffz(empty);

        case DYNAMICS_SFP :
            return factory.createDynamicsSfp(empty);

        case DYNAMICS_SFPP :
            return factory.createDynamicsSfpp(empty);

        case DYNAMICS_SFZ :
            return factory.createDynamicsSfz(empty);
        }

        logger.severe("Unsupported dynamics shape:" + shape);

        return null;
    }

    //-------------//
    // getTypeName //
    //-------------//
    /**
     * Report the name for the note type
     *
     * @param note the note whose type name is needed
     * @return proper note type name
     */
    public static String getNoteTypeName (Note note)
    {
        // Since quarter is at index 6 in noteTypeNames, use 2**6 = 64
        int dur = (64 * note.getNoteDuration()) / Note.QUARTER_DURATION;
        int index = (int) Math.rint(Math.log(dur) / Math.log(2));

        return noteTypeNames[index];
    }

    //-------------------//
    // getOrnamentObject //
    //-------------------//
    public static JAXBElement<?> getOrnamentObject (Shape shape)
    {
        //      (((trill-mark | turn | delayed-turn | shake |
        //         wavy-line | mordent | inverted-mordent |
        //         schleifer | tremolo | other-ornament),
        //         accidental-mark*)*)>
        ObjectFactory factory = new ObjectFactory();

        switch (shape) {
        case INVERTED_MORDENT :
            return factory.createOrnamentsInvertedMordent(
                factory.createMordent());

        case MORDENT :
            return factory.createOrnamentsMordent(factory.createMordent());

        case TR :
            return factory.createOrnamentsTrillMark(
                factory.createEmptyTrillSound());

        case TURN :
            return factory.createOrnamentsTurn(factory.createEmptyTrillSound());
        }

        logger.severe("Unsupported ornament shape:" + shape);

        return null;
    }

    //-------------//
    // getSyllabic //
    //-------------//
    public static Syllabic getSyllabic (LyricsItem.SyllabicType type)
    {
        return Syllabic.valueOf(type.toString());
    }

    //------------------//
    // accidentalTextOf //
    //------------------//
    public static AccidentalText accidentalTextOf (Shape shape)
    {
        ///sharp, natural, flat, double-sharp, sharp-sharp, flat-flat
        // But no double-flat ???
        if (shape == Shape.DOUBLE_FLAT) {
            return AccidentalText.FLAT_FLAT;
        } else {
            return AccidentalText.valueOf(shape.toString());
        }
    }

    //------------//
    // barStyleOf //
    //------------//
    /**
     * Report the MusicXML bar style for a recognized Barline shape
     *
     * @param shape the barline shape
     * @return the bar style
     */
    public static BarStyle barStyleOf (Shape shape)
    {
        //      Bar-style contains style information. Choices are
        //      regular, dotted, dashed, heavy, light-light,
        //      light-heavy, heavy-light, heavy-heavy, and none.
        switch (shape) {
        case THIN_BARLINE :
        case PART_DEFINING_BARLINE :
            return BarStyle.REGULAR; //"light" ???

        case DOUBLE_BARLINE :
            return BarStyle.LIGHT_LIGHT;

        case FINAL_BARLINE :
        case RIGHT_REPEAT_SIGN :
            return BarStyle.LIGHT_HEAVY;

        case REVERSE_FINAL_BARLINE :
        case LEFT_REPEAT_SIGN :
            return BarStyle.HEAVY_LIGHT;

        case BACK_TO_BACK_REPEAT_SIGN :
            return BarStyle.HEAVY_HEAVY; //"heavy-heavy"; ???
        }

        return BarStyle.NONE; // TO BE CHECKED ???
    }

    //---------------//
    // createDecimal //
    //---------------//
    public static BigDecimal createDecimal (double val)
    {
        return new BigDecimal("" + val);
    }

    //--------//
    // stepOf //
    //--------//
    /**
     * Convert from Audiveris Step type to Proxymusic Step type
     * @param step Audiveris enum step
     * @return Proxymusic enum step
     */
    public static Step stepOf (omr.score.entity.Note.Step step)
    {
        return Step.fromValue(step.toString());
    }

    //----------//
    // toTenths //
    //----------//
    /**
     * Convert a value expressed in units to a string value expressed in tenths
     *
     * @param units the number of units
     * @return the number of tenths as a string
     */
    public static BigDecimal toTenths (double units)
    {
        // Divide by 1.6 with rounding to nearest integer value
        return new BigDecimal("" + (int) Math.rint(units / 1.6));
    }

    //-----//
    // yOf //
    //-----//
    /**
     * Report the musicXML Y value of a SystemPoint ordinate.
     *
     * @param units the system-based ordinate (in units)
     * @param staff the related staff
     * @return the upward-oriented ordinate wrt to staff top line (in tenths)
     */
    public static BigDecimal yOf (double units,
                                  Staff  staff)
    {
        return toTenths(
            staff.getPageTopLeft().y - staff.getSystem().getTopLeft().y -
            units);
    }

    //-----//
    // yOf //
    //-----//
    /**
     * Report the musicXML Y value of a SystemPoint. This method is safer than
     * the other one which simply accepts a (detyped) double ordinate.
     *
     * @param point the system-based point
     * @param staff the related staff
     * @return the upward-oriented ordinate wrt to staff top line (in tenths)
     */
    public static BigDecimal yOf (SystemPoint point,
                                  Staff       staff)
    {
        return toTenths(
            staff.getPageTopLeft().y - staff.getSystem().getTopLeft().y -
            point.y);
    }
}
