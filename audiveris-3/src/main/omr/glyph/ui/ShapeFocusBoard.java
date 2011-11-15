//----------------------------------------------------------------------------//
//                                                                            //
//                       S h a p e F o c u s B o a r d                        //
//                                                                            //
//----------------------------------------------------------------------------//
// <editor-fold defaultstate="collapsed" desc="hdr">                          //
//  Copyright (C) Herve Bitteur 2000-2010. All rights reserved.               //
//  This software is released under the GNU General Public License.           //
//  Goto http://kenai.com/projects/audiveris to report bugs or suggestions.   //
//----------------------------------------------------------------------------//
// </editor-fold>
package omr.glyph.ui;

import omr.constant.Constant;
import omr.constant.ConstantSet;

import omr.glyph.GlyphRegression;
import omr.glyph.Shape;
import omr.glyph.ShapeRange;
import omr.glyph.facets.Glyph;

import omr.log.Logger;

import omr.math.LinearEvaluator.Printer;

import omr.selection.GlyphEvent;
import omr.selection.GlyphIdEvent;
import omr.selection.SelectionHint;
import omr.selection.UserEvent;

import omr.sheet.Sheet;

import omr.ui.Board;
import omr.ui.field.SpinnerUtilities;
import static omr.ui.field.SpinnerUtilities.*;
import omr.ui.util.Panel;

import omr.util.Implement;

import com.jgoodies.forms.builder.*;
import com.jgoodies.forms.layout.*;

import org.bushe.swing.event.EventSubscriber;

import java.awt.event.*;
import java.util.*;

import javax.swing.*;
import javax.swing.event.*;

/**
 * Class <code>ShapeFocusBoard</code> handles a user iteration within a
 * collection of glyphs. The collection may be built from glyphs of a given
 * shape, or from glyphs similar to a given glyph, etc.
 *
 * @author Hervé Bitteur
 */
public class ShapeFocusBoard
    extends Board
{
    //~ Static fields/initializers ---------------------------------------------

    /** Specific application parameters */
    private static final Constants constants = new Constants();

    /** Usual logger utility */
    private static final Logger logger = Logger.getLogger(
        ShapeFocusBoard.class);

    /** Events this board is interested in */
    private static final Collection<Class<?extends UserEvent>> eventClasses;

    static {
        eventClasses = new ArrayList<Class<?extends UserEvent>>();
        eventClasses.add(GlyphEvent.class);
    }

    //~ Enumerations -----------------------------------------------------------

    /** Filter on which symbols should be displayed */
    private static enum Filter {
        //~ Enumeration constant initializers ----------------------------------


        /** Display all symbols */
        ALL,
        /** Display only known symbols */
        KNOWN, 
        /** Display only unknown symbols */
        UNKNOWN, 
        /** Display only translated symbols */
        TRANSLATED, 
        /** Display only untranslated symbols */
        UNTRANSLATED;
    }

    //~ Instance fields --------------------------------------------------------

    private final Sheet sheet;

    /** Browser on the collection of glyphs */
    private Browser browser = new Browser();

    /** Button to select the shape focus */
    private JButton selectButton = new JButton();

    /** Filter for known / unknown symbol display */
    private JComboBox filterButton = new JComboBox(Filter.values());

    /** Popup menu to allow shape selection */
    private JPopupMenu pm = new JPopupMenu();

    //~ Constructors -----------------------------------------------------------

    //-----------------//
    // ShapeFocusBoard //
    //-----------------//
    /**
     * Create the instance to handle the shape focus, with pointers to needed
     * companions
     *
     * @param sheet the related sheet
     * @param controller the related glyph controller
     * @param filterListener the action linked to filter button
     */
    public ShapeFocusBoard (Sheet            sheet,
                            GlyphsController controller,
                            ActionListener   filterListener)
    {
        super(
            sheet.getRadix() + "-ShapeFocusBoard",
            "Focus",
            controller.getLag().getSelectionService(),
            eventClasses);

        this.sheet = sheet;

        // Tool Tips
        selectButton.setToolTipText("Select candidate shape");
        selectButton.setHorizontalAlignment(SwingConstants.LEFT);
        selectButton.addActionListener(
            new ActionListener() {
                    public void actionPerformed (ActionEvent e)
                    {
                        pm.show(
                            selectButton,
                            selectButton.getX(),
                            selectButton.getY());
                    }
                });

        // Filter
        filterButton.addActionListener(filterListener);
        filterButton.setToolTipText(
            "Select displayed glyphs according to their current state");

        // Popup menu for shape selection
        JMenuItem noFocus = new JMenuItem("No Focus");
        noFocus.setToolTipText("Cancel any focus");
        noFocus.addActionListener(
            new ActionListener() {
                    public void actionPerformed (ActionEvent e)
                    {
                        setCurrentShape(null);
                    }
                });
        pm.add(noFocus);
        ShapeRange.addShapeItems(
            pm,
            new ActionListener() {
                    public void actionPerformed (ActionEvent e)
                    {
                        JMenuItem source = (JMenuItem) e.getSource();
                        setCurrentShape(Shape.valueOf(source.getText()));
                    }
                });

        defineLayout();

        // Initially, no focus
        setCurrentShape(null);
    }

    //~ Methods ----------------------------------------------------------------

    //-----------------//
    // setCurrentShape //
    //-----------------//
    /**
     * Define the new current shape
     *
     * @param currentShape the shape to be considered as current
     */
    public void setCurrentShape (Shape currentShape)
    {
        browser.resetIds();

        if (currentShape != null) {
            // Update the shape button
            selectButton.setText(currentShape.toString());
            selectButton.setIcon(currentShape.getDecoratedSymbol());

            // Count the number of glyphs assigned to current shape
            for (Glyph glyph : sheet.getActiveGlyphs()) {
                if (glyph.getShape() == currentShape) {
                    browser.addId(glyph.getId());
                }
            }

            expand(); // Expand this board if so needed
        } else {
            // Void the shape button
            selectButton.setText("- No Focus -");
            selectButton.setIcon(null);
        }

        browser.refresh();
    }

    //-------------//
    // isDisplayed //
    //-------------//
    /**
     * Report whether the glyph at hand is to be displayed, according to the
     * current filter
     * @param glyph the glyph at hande
     * @return true if to be displayed
     */
    public boolean isDisplayed (Glyph glyph)
    {
        switch ((Filter) filterButton.getSelectedItem()) {
        case KNOWN :
            return glyph.isKnown();

        case UNKNOWN :
            return !glyph.isKnown();

        case TRANSLATED :
            return glyph.isKnown() && glyph.isTranslated();

        case UNTRANSLATED :
            return glyph.isKnown() && !glyph.isTranslated();

        default :
        case ALL :
            return true;
        }
    }

    //-----------------//
    // setSimilarGlyph //
    //-----------------//
    /**
     * Define the glyphs collection as all glyphs whose physical appearance is
     * "similar" to the appearance of the provided glyph example
     * @param example the provided example
     */
    public void setSimilarGlyph (Glyph example)
    {
        browser.resetIds();

        if (example != null) {
            GlyphRegression  evaluator = GlyphRegression.getInstance();
            double[]         pattern = GlyphRegression.feedInput(example, null);
            List<DistIdPair> pairs = new ArrayList<DistIdPair>();

            // Retrieve the glyphs similar to the example
            for (Glyph glyph : sheet.getActiveGlyphs()) {
                double dist = evaluator.measureDistance(glyph, pattern);
                pairs.add(new DistIdPair(dist, glyph.getId()));
            }

            Collections.sort(pairs, DistIdPair.distComparator);

            for (DistIdPair pair : pairs) {
                browser.addId(pair.id);
            }

            // To get a detailed table of the distances (debugging)
            if (constants.printDistances.getValue()) {
                Printer printer = evaluator.getEngine().new Printer(11);
                String  indent = "                  ";
                System.out.println(indent + printer.getDefaults());
                System.out.println(indent + printer.getNames());
                System.out.println(indent + printer.getDashes());

                for (DistIdPair pair : pairs) {
                    Glyph    glyph = sheet.getVerticalsController()
                                          .getGlyphById(pair.id);
                    double[] gPat = GlyphRegression.feedInput(glyph, null);
                    Shape    shape = glyph.getShape();
                    System.out.printf(
                        "%18s",
                        (shape != null) ? shape.toString() : "");
                    System.out.println(printer.getDeltas(gPat, pattern));
                    System.out.printf("g#%04d d:%9f", pair.id, pair.dist);
                    System.out.println(
                        printer.getWeightedDeltas(gPat, pattern));
                }
            }

            // Update the shape button
            selectButton.setText("Glyphs similar to #" + example.getId());
            selectButton.setIcon(null);
            
            // Expand this board if so needed
            expand();
        } else {
            // Void the shape button
            selectButton.setText("- No Focus -");
            selectButton.setIcon(null);
        }

        browser.refresh();
    }

    //---------//
    // onEvent //
    //---------//
    /**
     * Notification about selection objects
     * We used to use it on a just modified glyph, to set the new shape focus
     * But this conflicts with the ability to browse a collection of similar
     * glyphs and assign them on the fly
     *
     * @param event the notified event
     */
    @Implement(EventSubscriber.class)
    public void onEvent (UserEvent event)
    {
        // Empty
    }

    //--------------//
    // defineLayout //
    //--------------//
    private void defineLayout ()
    {
        final String buttonWidth = Panel.getButtonWidth();
        final String fieldInterval = Panel.getFieldInterval();
        final String fieldInterline = Panel.getFieldInterline();

        FormLayout   layout = new FormLayout(
            buttonWidth + "," + fieldInterval + "," + buttonWidth + "," +
            fieldInterval + "," + buttonWidth + "," + fieldInterval + "," +
            buttonWidth,
            "pref," + fieldInterline + "," + "pref");

        PanelBuilder builder = new PanelBuilder(layout, getBody());
        builder.setDefaultDialogBorder();

        CellConstraints cst = new CellConstraints();

        int             r = 1; // --------------------------------
        builder.add(selectButton, cst.xyw(3, r, 5));

        r += 2; // --------------------------------
        builder.add(filterButton, cst.xy(1, r));

        builder.add(browser.count, cst.xy(5, r));
        builder.add(browser.spinner, cst.xy(7, r));
    }

    //~ Inner Classes ----------------------------------------------------------

    //------------//
    // DistIdPair //
    //------------//
    /**
     * Needed to sort glyphs id according to their distance
     */
    private static class DistIdPair
    {
        //~ Static fields/initializers -----------------------------------------

        private static final Comparator<DistIdPair> distComparator = new Comparator<DistIdPair>() {
            public int compare (DistIdPair o1,
                                DistIdPair o2)
            {
                return Double.compare(o1.dist, o2.dist);
            }
        };


        //~ Instance fields ----------------------------------------------------

        final double dist;
        final int    id;

        //~ Constructors -------------------------------------------------------

        public DistIdPair (double dist,
                           int    id)
        {
            this.dist = dist;
            this.id = id;
        }

        //~ Methods ------------------------------------------------------------

        @Override
        public String toString ()
        {
            return "dist:" + dist + " glyph#" + id;
        }
    }

    //---------//
    // Browser //
    //---------//
    private class Browser
        implements ChangeListener
    {
        //~ Instance fields ----------------------------------------------------

        // Spinner on these glyphs
        ArrayList<Integer> ids = new ArrayList<Integer>();

        // Number of glyphs
        JLabel   count = new JLabel("", SwingConstants.RIGHT);
        JSpinner spinner = new JSpinner(new SpinnerListModel());

        //~ Constructors -------------------------------------------------------

        //---------//
        // Browser //
        //---------//
        public Browser ()
        {
            resetIds();
            spinner.addChangeListener(this);
            SpinnerUtilities.setList(spinner, ids);
            refresh();
        }

        //~ Methods ------------------------------------------------------------

        //-------//
        // addId //
        //-------//
        public void addId (int id)
        {
            ids.add(id);
        }

        //---------//
        // refresh //
        //---------//
        public void refresh ()
        {
            if (ids.size() > 1) { // To skip first NO_VALUE item
                count.setText(0 + "/" + (ids.size() - 1));
                spinner.setEnabled(true);
            } else {
                count.setText("");
                spinner.setEnabled(false);
            }

            spinner.setValue(NO_VALUE);
        }

        //----------//
        // resetIds //
        //----------//
        public void resetIds ()
        {
            ids.clear();
            ids.add(NO_VALUE);
        }

        //--------------//
        // stateChanged //
        //--------------//
        @Implement(ChangeListener.class)
        public void stateChanged (ChangeEvent e)
        {
            int id = (Integer) spinner.getValue();

            int index = ids.indexOf(id);
            count.setText(index + "/" + (ids.size() - 1));

            if (id != NO_VALUE) {
                selectionService.publish(
                    new GlyphIdEvent(this, SelectionHint.GLYPH_INIT, null, id));
            }
        }
    }

    //-----------//
    // Constants //
    //-----------//
    private static final class Constants
        extends ConstantSet
    {
        //~ Instance fields ----------------------------------------------------

        Constant.Boolean printDistances = new Constant.Boolean(
            false,
            "Should we print out distance details when looking for similar glyphs?");
    }
}
