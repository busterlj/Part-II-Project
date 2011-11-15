//----------------------------------------------------------------------------//
//                                                                            //
//                            S h a p e R a n g e                             //
//                                                                            //
//----------------------------------------------------------------------------//
// <editor-fold defaultstate="collapsed" desc="hdr">                          //
//  Copyright (C) Herve Bitteur 2000-2010. All rights reserved.               //
//  This software is released under the GNU General Public License.           //
//  Goto http://kenai.com/projects/audiveris to report bugs or suggestions.   //
//----------------------------------------------------------------------------//
// </editor-fold>
package omr.glyph;

import omr.constant.Constant;
import static omr.glyph.Shape.*;

import java.awt.Color;
import java.awt.event.ActionListener;
import java.lang.reflect.Field;
import java.util.*;

import javax.swing.*;

/**
 * Class {@code ShapeRange} defines a set of related shapes, for example
 * the "Rests" range gathers all rest shapes from MULTI_REST down to
 * ONE_HUNDRED_TWENTY_EIGHTH_REST.
 *
 * <p>It handles additional properties over a simple EnumSet, especially
 * assigned colors and its automatic insertion in shape menu hierarchy.
 * So don't remove any of these sets, unless you know what you are doing.
 */
public class ShapeRange
{
    //~ Static fields/initializers ---------------------------------------------

    // Predefined instances of ShapeRange. Double-check before removing any one.
    //
    public static final ShapeRange Accidentals = new ShapeRange(
        SHARP,
        EnumSet.range(FLAT, DOUBLE_FLAT));
    public static final ShapeRange Articulations = new ShapeRange(
        ARPEGGIATO,
        EnumSet.range(ACCENT, STRONG_ACCENT),
        STACCATO,
        ARPEGGIATO);
    public static final ShapeRange Barlines = new ShapeRange(
        LEFT_REPEAT_SIGN,
        EnumSet.range(PART_DEFINING_BARLINE, BACK_TO_BACK_REPEAT_SIGN));
    public static final ShapeRange Beams = new ShapeRange(
        BEAM,
        EnumSet.range(BEAM, BEAM_HOOK),
        COMBINING_STEM);
    public static final ShapeRange Clefs = new ShapeRange(
        G_CLEF,
        EnumSet.range(G_CLEF, PERCUSSION_CLEF));
    public static final ShapeRange Dynamics = new ShapeRange(
        DYNAMICS_F,
        EnumSet.range(DYNAMICS_CHAR_M, DECRESCENDO));
    public static final ShapeRange Flags = new ShapeRange(
        COMBINING_FLAG_1,
        EnumSet.range(COMBINING_FLAG_1, COMBINING_FLAG_5_UP));
    public static final ShapeRange HeadAndFlags = new ShapeRange(
        HEAD_AND_FLAG_1,
        EnumSet.range(HEAD_AND_FLAG_1, HEAD_AND_FLAG_5_UP));
    public static final ShapeRange NoteHeads = new ShapeRange(
        NOTEHEAD_BLACK,
        EnumSet.range(VOID_NOTEHEAD, NOTEHEAD_BLACK_3));
    public static final ShapeRange Markers = new ShapeRange(
        CODA,
        EnumSet.range(DAL_SEGNO, BRACKET));
    public static final ShapeRange Notes = new ShapeRange(
        BREVE,
        EnumSet.range(BREVE, WHOLE_NOTE_3));
    public static final ShapeRange Ornaments = new ShapeRange(
        MORDENT,
        EnumSet.range(GRACE_NOTE_SLASH, INVERTED_MORDENT));
    public static final ShapeRange Rests = new ShapeRange(
        QUARTER_REST,
        Arrays.asList(
            LONG_REST,
            BREVE_REST,
            WHOLE_REST,
            HALF_REST,
            QUARTER_REST,
            OLD_QUARTER_REST,
            EIGHTH_REST,
            SIXTEENTH_REST,
            THIRTY_SECOND_REST,
            SIXTY_FOURTH_REST,
            ONE_HUNDRED_TWENTY_EIGHTH_REST));
    public static final ShapeRange Times = new ShapeRange(
        TIME_FOUR_FOUR,
        EnumSet.range(TIME_ZERO, CUT_TIME),
        CUSTOM_TIME_SIGNATURE);

    /** A bag of miscellaneous shapes */
    public static final ShapeRange Others = new ShapeRange(
        PEDAL_MARK,
        EnumSet.of(
            SLUR,
            OTTAVA_ALTA,
            OTTAVA_BASSA,
            PEDAL_MARK,
            PEDAL_UP_MARK,
            TUPLET_THREE,
            TUPLET_SIX));

    //
    public static final ShapeRange Physicals = new ShapeRange(
        EnumSet.range(CLUTTER, TEXT));
    public static final ShapeRange Logicals = new ShapeRange(
        EnumSet.range(REPEAT_DOTS, ENDING));

    static {
        for (Field field : ShapeRange.class.getDeclaredFields()) {
            if (field.getType() == ShapeRange.class) {
                try {
                    ShapeRange range = (ShapeRange) field.get(null);
                    range.setName(field.getName());

                    // Create shape color for all contained shapes
                    for (Shape shape : range.shapes) {
                        shape.createShapeColor(range.getColor());
                    }
                } catch (IllegalAccessException ex) {
                    ex.printStackTrace();
                }
            }
        }

        WHOLE_OR_HALF_REST.createShapeColor(Rests.getColor());
    }

    // =========================================================================
    // Below are EnumSet instances, used programmatically.
    // They do not lead to shape submenus as the ShapeRange instances do.
    // =========================================================================

    /** All physical shapes */
    public static final EnumSet<Shape> allSymbols = EnumSet.range(
        Shape.values()[0],
        LAST_PHYSICAL_SHAPE);

    /** Symbols that can be attached to a stem */
    public static final EnumSet<Shape> StemSymbols = EnumSet.noneOf(
        Shape.class);

    static {
        StemSymbols.add(BEAM);
        StemSymbols.add(BEAM_2);
        StemSymbols.add(BEAM_3);
        StemSymbols.add(BEAM_HOOK);

        StemSymbols.addAll(NoteHeads.getShapes());
        StemSymbols.addAll(Flags.getShapes());
        StemSymbols.addAll(HeadAndFlags.getShapes());
    }

    /** Pedals */
    public static final EnumSet<Shape> Pedals = EnumSet.range(
        PEDAL_MARK,
        PEDAL_UP_MARK);

    /** Tuplets */
    public static final EnumSet<Shape> Tuplets = EnumSet.range(
        TUPLET_THREE,
        TUPLET_SIX);

    /** Specific single symbol for part of time signature (such as 4) */
    public static final EnumSet<Shape> PartialTimes = EnumSet.range(
        TIME_ZERO,
        TIME_SIXTEEN);

    /** Specific multi-symbol for full time signature (such as 4/4 */
    public static final EnumSet<Shape> FullTimes = EnumSet.range(
        TIME_FOUR_FOUR,
        CUT_TIME);

    static {
        FullTimes.add(CUSTOM_TIME_SIGNATURE);
    }

    /** All variants of dot */
    public static final EnumSet<Shape> Dots = EnumSet.of(
        DOT,
        COMBINING_AUGMENTATION_DOT,
        STACCATO,
        REPEAT_DOTS);

    /** All sorts of F clefs */
    public static final EnumSet<Shape> BassClefs = EnumSet.of(
        F_CLEF,
        F_CLEF_OTTAVA_ALTA,
        F_CLEF_OTTAVA_BASSA);

    //~ Instance fields --------------------------------------------------------

    /** Name of the range */
    private String name;

    /** Underlying shapes */
    private final EnumSet<Shape> shapes;

    /** Specific sequence of shapes, if any */
    private final List<Shape> sortedShapes;

    /** The representative shape for this range */
    private final Shape rep;

    /** Current color */
    private Color color;

    /** Related permanent color */
    private Constant.Color constantColor;

    //~ Constructors -----------------------------------------------------------

    /**
     * Creates a new ShapeRange object from a specified sequence of shapes
     *
     * @param rep the representative shape
     * @param sortedShapes specific sequence of shapes
     */
    public ShapeRange (Shape       rep,
                       List<Shape> sortedShapes)
    {
        this.rep = rep;
        shapes = EnumSet.noneOf(Shape.class);

        for (Shape shape : sortedShapes) {
            shapes.add(shape);
        }

        this.sortedShapes = new ArrayList<Shape>(sortedShapes);
    }

    /**
     * Creates a new ShapeRange object.
     *
     * @param rep the representative shape
     * @param shapes the set of shapes defining the range
     * @param addedShapes shapes added to the initial range
     */
    public ShapeRange (Shape          rep,
                       EnumSet<Shape> shapes,
                       Shape... addedShapes)
    {
        this.rep = rep;
        this.shapes = shapes;

        for (Shape shape : addedShapes) {
            shapes.add(shape);
        }

        sortedShapes = null;
    }

    /**
     * Creates a new ShapeRange object.
     *
     * @param shapes the set of shapes defining the range
     * @param addedShapes shapes added to the initial range
     */
    public ShapeRange (EnumSet<Shape> shapes,
                       Shape... addedShapes)
    {
        this(null, shapes, addedShapes);
    }

    //~ Methods ----------------------------------------------------------------

    //----------//
    // setColor //
    //----------//
    /**
     * Assign a display color to the shape range
     *
     * @param color the display color
     */
    public void setColor (Color color)
    {
        this.color = color;
    }

    //----------//
    // getColor //
    //----------//
    /**
     * Report the color currently assigned to the range, if any
     *
     * @return the related color, or null
     */
    public Color getColor ()
    {
        return color;
    }

    //------------------//
    // setConstantColor //
    //------------------//
    /**
     * Define a specific color for the range
     *
     * @param color the specified color
     */
    public void setConstantColor (Color color)
    {
        constantColor.setValue(color);
        setColor(color);
    }

    //----------//
    // getRange //
    //----------//
    public static ShapeRange getRange (String name)
    {
        return Ranges.map.get(name);
    }

    //-----------//
    // getRanges //
    //-----------//
    public static List<ShapeRange> getRanges ()
    {
        return Ranges.rangeList;
    }

    //--------//
    // getRep //
    //--------//
    /**
     * Report the representative shape of the range, if any
     * @return the rep shape, or null
     */
    public Shape getRep ()
    {
        return rep;
    }

    //-----------//
    // getShapes //
    //-----------//
    /**
     * Exports the set of shapes in the range
     *
     * @return the proper enum set
     */
    public EnumSet<Shape> getShapes ()
    {
        return shapes;
    }

    //-----------------//
    // getSortedShapes //
    //-----------------//
    /**
     * Exports the sorted collection of shapes
     *
     * @return the proper enum set
     */
    public List<Shape> getSortedShapes ()
    {
        if (sortedShapes != null) {
            return sortedShapes;
        } else {
            return new ArrayList<Shape>(shapes);
        }
    }

    //------------------//
    // addAllRangeItems //
    //------------------//
    /**
     * Populate the given menu with a hierarchy of all ranges defined in
     * ShapeRange classs
     *
     * @param top the JComponent to populate (typically a JMenu or a
     * JPopupMenu)
     * @param listener the listener for notification of user selection
     */
    public static void addAllRangeItems (JComponent     top,
                                         ActionListener listener)
    {
        // All ranges of glyph shapes
        for (Field field : ShapeRange.class.getDeclaredFields()) {
            if (field.getType() == ShapeRange.class) {
                JMenuItem  menuItem = new JMenuItem(field.getName());
                ShapeRange range = valueOf(field.getName());
                addColoredItem(top, menuItem, range.getColor());
                menuItem.addActionListener(listener);
            }
        }
    }

    //---------//
    // getName //
    //---------//
    /**
     * Report the name of the range
     *
     * @return the range name
     */
    public String getName ()
    {
        return name;
    }

    //---------------//
    // addRangeItems //
    //---------------//
    /**
     * Populate the given menu with a list of all shapes that belong to the
     * given range
     *
     * @param range the range for which shape menu items must be buit
     * @param top the JComponent to populate (typically a JMenu or a JPopupMenu)
     * @param listener the listener for notification of user selection
     */
    public static void addRangeItems (ShapeRange     range,
                                      JComponent     top,
                                      ActionListener listener)
    {
        // All shapes in the given range
        for (Shape shape : range.shapes) {
            JMenuItem menuItem = new JMenuItem(
                shape.toString(),
                shape.getDecoratedSymbol());
            addColoredItem(top, menuItem, shape.getColor());

            menuItem.setToolTipText(shape.getDescription());
            menuItem.addActionListener(listener);
        }
    }

    //---------------//
    // addShapeItems //
    //---------------//
    /**
     * Populate the given menu with a hierarchy of all shapes, organized by
     * defined ranges.
     *
     * @param top the JComponent to populate (typically a JMenu or a JPopupMenu)
     * @param listener the listener for notification of user selection
     */
    public static void addShapeItems (JComponent     top,
                                      ActionListener listener)
    {
        // All ranges of glyph shapes
        for (Field field : ShapeRange.class.getDeclaredFields()) {
            if (field.getType() == ShapeRange.class) {
                ShapeRange range = ShapeRange.valueOf(field.getName());
                JMenu      menu = new JMenu(field.getName());

                if (range.rep != null) {
                    menu.setIcon(range.rep.getDecoratedSymbol());
                }

                addColoredItem(top, menu, Color.black);

                // Add menu items for this range
                addRangeItems(range, menu, listener);
            }
        }
    }

    //----------//
    // contains //
    //----------//
    /**
     * Convenient method to check if encapsulated shapes set does contain
     * the provided object
     *
     * @param shape the Shape object to check for inclusion
     * @return true if contained, false otherwise
     */
    public boolean contains (Shape shape)
    {
        return shapes.contains(shape);
    }

    //---------//
    // valueOf //
    //---------//
    /**
     * Retrieve a range knowing its name (just like an enumeration)
     *
     * @param str the provided range name
     * @return the range found, or null otherwise
     */
    public static ShapeRange valueOf (String str)
    {
        return Ranges.map.get(str);
    }

    //---------//
    // setName //
    //---------//
    private void setName (String name)
    {
        this.name = name;
        constantColor = new Constant.Color(
            getClass().getName(),
            name + ".color",
            "#000000",
            "Color code for range " + name);

        // Assign the range display color
        if (!constantColor.isSourceValue()) {
            setColor(constantColor.getValue());
        } else {
            // Use the default color
            setColor(Color.black);
        }
    }

    //----------------//
    // addColoredItem //
    //----------------//
    private static void addColoredItem (JComponent top,
                                        JMenuItem  item,
                                        Color      color)
    {
        if (color != null) {
            item.setForeground(color);
        } else {
            item.setForeground(Color.black);
        }

        top.add(item);
    }

    //~ Inner Classes ----------------------------------------------------------

    //--------//
    // Ranges //
    //--------//
    /** Build the range map in a lazy way */
    private static class Ranges
    {
        //~ Static fields/initializers -----------------------------------------

        static Map<String, ShapeRange> map = new HashMap<String, ShapeRange>();
        static List<ShapeRange>        rangeList = new ArrayList<ShapeRange>();

        static {
            for (Field field : ShapeRange.class.getDeclaredFields()) {
                if (field.getType() == ShapeRange.class) {
                    try {
                        ShapeRange range = (ShapeRange) field.get(null);
                        map.put(field.getName(), range);
                        rangeList.add(range);
                    } catch (IllegalAccessException ex) {
                        ex.printStackTrace();
                    }
                }
            }
        }
    }
}
