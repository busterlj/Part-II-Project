//----------------------------------------------------------------------------//
//                                                                            //
//                          G l y p h B r o w s e r                           //
//                                                                            //
//----------------------------------------------------------------------------//
// <editor-fold defaultstate="collapsed" desc="hdr">                          //
//  Copyright (C) Herve Bitteur 2000-2010. All rights reserved.               //
//  This software is released under the GNU General Public License.           //
//  Goto http://kenai.com/projects/audiveris to report bugs or suggestions.   //
//----------------------------------------------------------------------------//
// </editor-fold>
package omr.glyph.ui;

import omr.WellKnowns;

import omr.constant.Constant;
import omr.constant.ConstantSet;

import omr.glyph.GlyphLag;
import omr.glyph.GlyphSection;
import omr.glyph.GlyphsModel;
import omr.glyph.facets.Glyph;

import omr.lag.VerticalOrientation;
import omr.lag.ui.ScrollLagView;

import omr.log.Logger;

import omr.selection.GlyphEvent;
import omr.selection.MouseMovement;
import omr.selection.SelectionHint;
import static omr.selection.SelectionHint.*;
import omr.selection.SelectionService;
import omr.selection.SheetLocationEvent;
import omr.selection.UserEvent;

import omr.ui.Board;
import omr.ui.field.LTextField;
import omr.ui.util.Panel;
import omr.ui.view.LogSlider;
import omr.ui.view.Rubber;
import omr.ui.view.Zoom;

import omr.util.BlackList;
import omr.util.Implement;

import com.jgoodies.forms.builder.PanelBuilder;
import com.jgoodies.forms.layout.CellConstraints;
import com.jgoodies.forms.layout.FormLayout;

import java.awt.*;
import java.awt.event.*;
import java.io.*;
import java.util.*;
import java.util.List;

import javax.swing.*;
import javax.swing.event.*;

/**
 * Class <code>GlyphBrowser</code> gathers a navigator to move between selected
 * glyphs, a glyph board for glyph details, and a display for graphical glyph
 * lag view. This is a (package private) companion of {@link GlyphVerifier}.
 *
 * @author Hervé Bitteur
 */
class GlyphBrowser
    implements ChangeListener
{
    //~ Static fields/initializers ---------------------------------------------

    /** Specific application parameters */
    private static final Constants constants = new Constants();

    /** Usual logger utility */
    private static final Logger logger = Logger.getLogger(GlyphBrowser.class);

    /**
     * Field constant <code>NO_INDEX</code> is a specific value {@value} to
     * indicate absence of index
     */
    private static final int NO_INDEX = -1;

    //~ Instance fields --------------------------------------------------------

    /** The concrete Swing component */
    private JPanel component = new JPanel();

    /** Reference of GlyphVerifier */
    private final GlyphVerifier verifier;

    /** Repository of known glyphs */
    private final GlyphRepository repository = GlyphRepository.getInstance();

    /** Size of the lag display */
    private Dimension modelSize;

    /** Contour of the lag display */
    private Rectangle modelRectangle;

    /** Population of glyphs file names */
    private List<String> names;

    /** Navigator instance to navigate through all glyphs names */
    private Navigator navigator;

    /** Left panel : navigator, glyphboard, evaluator */
    private JPanel leftPanel;

    /** Composite display (view + zoom slider) */
    private Display display;

    /** Basic (sheet) event service */
    private SelectionService locationService;

    /** Hosting GlyphLag */
    private GlyphLag tLag;

    /** Basic glyph model */
    private GlyphsController controller;

    /** The lag view */
    private GlyphLagView view;

    /** Glyph board with ability to delete a training glyph */
    private GlyphBoard glyphBoard;

    //~ Constructors -----------------------------------------------------------

    //--------------//
    // GlyphBrowser //
    //--------------//
    /**
     * Create an instance, with back-reference to GlyphVerifier
     *
     * @param verifier ref back to verifier
     */
    public GlyphBrowser (GlyphVerifier verifier)
    {
        this.verifier = verifier;

        // Layout
        component.setLayout(new BorderLayout());
        resetBrowser();
    }

    //~ Methods ----------------------------------------------------------------

    //--------------//
    // getComponent //
    //--------------//
    /**
     * Report the UI component
     *
     * @return the concrete component
     */
    public JPanel getComponent ()
    {
        return component;
    }

    //----------------//
    // loadGlyphNames //
    //----------------//
    /**
     * Programmatic use of Load action in Navigator : load the glyph names as
     * selected, and focus on first glyph
     */
    public void loadGlyphNames ()
    {
        navigator.loadAction.actionPerformed(null);
    }

    //--------------//
    // stateChanged //
    //--------------//
    @Implement(ChangeListener.class)
    public void stateChanged (ChangeEvent e)
    {
        int selNb = verifier.getGlyphNames().length;
        navigator.loadAction.setEnabled(selNb > 0);
    }

    //----------------//
    // buildLeftPanel //
    //----------------//
    /**
     * Build a panel composed vertically of a Navigator, a GlyphBoard and an
     * EvaluationBoard
     *
     * @return the UI component, ready to be inserted in Swing hierarchy
     */
    private JPanel buildLeftPanel ()
    {
        navigator = new Navigator();

        // Specific glyph board
        glyphBoard = new MyGlyphBoard(controller);

        glyphBoard.connect();
        glyphBoard.getDeassignAction()
                  .setEnabled(false);

        // Passive evaluation board
        EvaluationBoard evalBoard = new EvaluationBoard(
            "Evaluation-PassiveBoard",
            controller);
        evalBoard.connect();

        // Layout
        FormLayout      layout = new FormLayout("pref", "pref,pref,pref");
        PanelBuilder    builder = new PanelBuilder(layout);
        CellConstraints cst = new CellConstraints();
        builder.setDefaultDialogBorder();

        builder.add(navigator.getComponent(), cst.xy(1, 1));
        builder.add(glyphBoard.getComponent(), cst.xy(1, 2));
        builder.add(evalBoard.getComponent(), cst.xy(1, 3));

        return builder.getPanel();
    }

    //-------------//
    // removeGlyph //
    //-------------//
    private void removeGlyph ()
    {
        int index = navigator.getIndex();

        if (index >= 0) {
            // Delete glyph designated by index
            String gName = names.get(index);
            Glyph  glyph = navigator.getGlyph(gName);

            // User confirmation is required ?
            if (constants.confirmDeletions.getValue()) {
                if (JOptionPane.showConfirmDialog(
                    component,
                    "Remove glyph '" + gName + "' ?") != JOptionPane.YES_OPTION) {
                    return;
                }
            }

            // Shrink names list
            names.remove(index);

            // Update model & display
            repository.removeGlyph(gName);

            for (GlyphSection section : glyph.getMembers()) {
                section.delete();
            }

            // Update the Glyph selector also !
            verifier.deleteGlyphName(gName);

            // Perform file deletion
            if (repository.isIcon(gName)) {
                new BlackList(WellKnowns.SYMBOLS_FOLDER).add(new File(gName));
            } else {
                File file = new File(repository.getSheetsFolder(), gName);
                new BlackList(file.getParentFile()).add(new File(gName));
            }

            logger.info("Removed " + gName);

            // Set new index ?
            if (index < names.size()) {
                navigator.setIndex(index, GLYPH_INIT); // Next
            } else {
                navigator.setIndex(index - 1, GLYPH_INIT); // Prev/None
            }
        } else {
            logger.warning("No selected glyph to remove!");
        }
    }

    //--------------//
    // resetBrowser //
    //--------------//
    private void resetBrowser ()
    {
        // Reset model
        tLag = new GlyphLag(
            "tLag",
            GlyphSection.class,
            new VerticalOrientation());

        locationService = new SelectionService();
        controller = new BasicController(tLag, locationService);

        // Reset left panel
        if (leftPanel != null) {
            component.remove(leftPanel);
        }

        leftPanel = buildLeftPanel();
        component.add(leftPanel, BorderLayout.WEST);

        // Reset display
        if (display != null) {
            component.remove(display);
        }

        display = new Display();
        component.add(display, BorderLayout.CENTER);

        // TODO: Check if all this is really needed ...
        component.invalidate();
        component.validate();
        component.repaint();
    }

    //~ Inner Classes ----------------------------------------------------------

    //-----------//
    // Constants //
    //-----------//
    private static final class Constants
        extends ConstantSet
    {
        //~ Instance fields ----------------------------------------------------

        Constant.Boolean confirmDeletions = new Constant.Boolean(
            true,
            "Should user confirm each glyph deletion" +
            " from training material");
    }

    //-----------------//
    // BasicController //
    //-----------------//
    /**
     * A very basic glyphs controller, with a sheet-less location service
     */
    private class BasicController
        extends GlyphsController
    {
        //~ Instance fields ----------------------------------------------------

        /** A specific location service, not tied to a sheet */
        private final SelectionService locationService;

        //~ Constructors -------------------------------------------------------

        public BasicController (GlyphLag         lag,
                                SelectionService locationService)
        {
            super(new BasicModel(lag));
            this.locationService = locationService;
        }

        //~ Methods ------------------------------------------------------------

        @Override
        public SelectionService getLocationService ()
        {
            return this.locationService;
        }
    }

    //------------//
    // BasicModel //
    //------------//
    /**
     * A very basic glyphs model, used to handle the deletion of glyphs
     */
    private class BasicModel
        extends GlyphsModel
    {
        //~ Constructors -------------------------------------------------------

        public BasicModel (GlyphLag lag)
        {
            super(null, lag, null);
        }

        //~ Methods ------------------------------------------------------------

        // Certainly not called ...
        @Override
        public void deassignGlyph (Glyph glyph)
        {
            removeGlyph();
        }
    }

    //----------------//
    // DeassignAction //
    //----------------//
    private class DeassignAction
        extends AbstractAction
    {
        //~ Constructors -------------------------------------------------------

        public DeassignAction ()
        {
            super("Remove");
            putValue(
                Action.SHORT_DESCRIPTION,
                "Remove that glyph from training material");
        }

        //~ Methods ------------------------------------------------------------

        @SuppressWarnings("unchecked")
        @Implement(ChangeListener.class)
        public void actionPerformed (ActionEvent e)
        {
            removeGlyph();
        }
    }

    //---------//
    // Display //
    //---------//
    private class Display
        extends JPanel
    {
        //~ Instance fields ----------------------------------------------------

        LogSlider     slider;
        Rubber        rubber;
        ScrollLagView slv;
        Zoom          zoom;

        //~ Constructors -------------------------------------------------------

        public Display ()
        {
            view = new MyView(controller);
            modelRectangle = new Rectangle();
            modelSize = new Dimension(0, 0);
            slider = new LogSlider(2, 5, LogSlider.VERTICAL, -3, 4, 0);
            zoom = new Zoom(slider, 1); // Default ratio set to 1
            rubber = new Rubber(view, zoom);
            rubber.setMouseMonitor(view);
            view.setZoom(zoom);
            view.setRubber(rubber);
            slv = new ScrollLagView(view);

            // Layout
            setLayout(new BorderLayout());
            add(slider, BorderLayout.WEST);
            add(slv.getComponent(), BorderLayout.CENTER);
        }
    }

    //--------------//
    // MyGlyphBoard //
    //--------------//
    private class MyGlyphBoard
        extends SymbolGlyphBoard
    {
        //~ Constructors -------------------------------------------------------

        public MyGlyphBoard (GlyphsController controller)
        {
            super("Browser-SymbolGlyphBoard", controller);
        }

        //~ Methods ------------------------------------------------------------

        @Override
        public Action getDeassignAction ()
        {
            if (deassignAction == null) {
                deassignAction = new DeassignAction();
            }

            return deassignAction;
        }
    }

    //--------//
    // MyView //
    //--------//
    private class MyView
        extends GlyphLagView
    {
        //~ Constructors -------------------------------------------------------

        public MyView (GlyphsController controller)
        {
            super(tLag, null, null, controller, null);
            setName("GlyphBrowser-View");

            subscribe();
        }

        //~ Methods ------------------------------------------------------------

        //---------------//
        // colorizeGlyph //
        //---------------//
        @Override
        public void colorizeGlyph (int   viewIndex,
                                   Glyph glyph)
        {
            colorizeGlyph(viewIndex, glyph, glyph.getColor());
        }

        //---------//
        // onEvent //
        //---------//
        /**
         * Call-back triggered from (local) selection objects
         *
         * @param event the notified event
         */
        @Override
        public void onEvent (UserEvent event)
        {
            try {
                // Ignore RELEASING
                if (event.movement == MouseMovement.RELEASING) {
                    return;
                }

                // Keep normal view behavior (rubber, etc...)
                super.onEvent(event);

                // Additional tasks
                if (event instanceof SheetLocationEvent) {
                    SheetLocationEvent sheetLocation = (SheetLocationEvent) event;

                    if (sheetLocation.hint == SelectionHint.LOCATION_INIT) {
                        Rectangle rect = sheetLocation.rectangle;

                        if ((rect != null) &&
                            (rect.width == 0) &&
                            (rect.height == 0)) {
                            // Look for pointed glyph
                            navigator.setIndex(
                                glyphLookup(rect),
                                sheetLocation.hint);
                        }
                    }
                } else if (event instanceof GlyphEvent) {
                    GlyphEvent glyphEvent = (GlyphEvent) event;

                    if (glyphEvent.hint == GLYPH_INIT) {
                        Glyph glyph = glyphEvent.getData();

                        // Display glyph contour
                        if (glyph != null) {
                            locationService.publish(
                                new SheetLocationEvent(
                                    this,
                                    glyphEvent.hint,
                                    null,
                                    glyph.getContourBox()));
                        }
                    }
                }
            } catch (Exception ex) {
                logger.warning(getClass().getName() + " onEvent error", ex);
            }
        }

        //-------------//
        // renderItems //
        //-------------//
        @Override
        public void renderItems (Graphics g)
        {
            // Mark the current glyph
            int index = navigator.getIndex();

            if (index >= 0) {
                String gName = names.get(index);
                Glyph  glyph = navigator.getGlyph(gName);
                g.setColor(Color.black);
                g.setXORMode(Color.darkGray);
                renderGlyphArea(glyph, g);
            }
        }

        //-------------//
        // glyphLookup //
        //-------------//
        /**
         * Lookup for a glyph that is pointed by rectangle location. This is a
         * very specific glyph lookup, for which we cannot rely on GlyphLag
         * usual features. So we simply browse through the collection of glyphs
         * (names).
         *
         * @param rect location (upper left corner)
         * @return index in names collection if found, NO_INDEX otherwise
         */
        private int glyphLookup (Rectangle rect)
        {
            int index = -1;

            for (String gName : names) {
                index++;

                if (repository.isLoaded(gName)) {
                    Glyph glyph = navigator.getGlyph(gName);

                    if (glyph.getLag() == tLag) {
                        for (GlyphSection section : glyph.getMembers()) {
                            // Swap x & y,  since this is a vertical lag
                            if (section.contains(rect.y, rect.x)) {
                                return index;
                            }
                        }
                    }
                }
            }

            return NO_INDEX; // Not found
        }
    }

    //-----------//
    // Navigator //
    //-----------//
    /**
     * Class <code>Navigator</code> handles the navigation through the
     * collection of glyphs (names)
     */
    private final class Navigator
        extends Board
    {
        //~ Instance fields ----------------------------------------------------

        /** Current index in names collection (NO_INDEX if none) */
        private int nameIndex = NO_INDEX;

        // Navigation actions & buttons
        LoadAction loadAction = new LoadAction();
        JButton    load = new JButton(loadAction);
        JButton    all = new JButton("All");
        JButton    next = new JButton("Next");
        JButton    prev = new JButton("Prev");
        LTextField nameField = new LTextField("", "File where glyph is stored");

        //~ Constructors -------------------------------------------------------

        //-----------//
        // Navigator //
        //-----------//
        Navigator ()
        {
            super("Glyph-Navigator", "Navigator", null, null);

            defineLayout();

            all.addActionListener(
                new ActionListener() {
                        public void actionPerformed (ActionEvent e)
                        {
                            // Load all (non icon) glyphs
                            int index = -1;

                            for (String gName : names) {
                                index++;

                                if (!repository.isIcon(gName)) {
                                    setIndex(index, GLYPH_INIT);
                                }
                            }

                            // Load & point to first icon
                            setIndex(0, GLYPH_INIT);
                        }
                    });

            prev.addActionListener(
                new ActionListener() {
                        public void actionPerformed (ActionEvent e)
                        {
                            setIndex(nameIndex - 1, GLYPH_INIT); // To prev
                        }
                    });

            next.addActionListener(
                new ActionListener() {
                        public void actionPerformed (ActionEvent e)
                        {
                            setIndex(nameIndex + 1, GLYPH_INIT); // To next
                        }
                    });

            load.setToolTipText("Load the selected glyphs");
            all.setToolTipText("Display all glyphs");
            prev.setToolTipText("Go to previous glyph");
            next.setToolTipText("Go to next glyph");

            loadAction.setEnabled(false);
            all.setEnabled(false);
            prev.setEnabled(false);
            next.setEnabled(false);
        }

        //~ Methods ------------------------------------------------------------

        //----------//
        // getIndex //
        //----------//
        /**
         * Report the current glyph index in the names collection
         *
         * @return the current index, which may be NO_INDEX
         */
        public final int getIndex ()
        {
            return nameIndex;
        }

        //----------//
        // getGlyph //
        //----------//
        public Glyph getGlyph (String gName)
        {
            Glyph glyph = repository.getGlyph(gName, null);

            if ((glyph != null) && (glyph.getLag() != tLag)) {
                glyph.setLag(tLag);

                for (GlyphSection section : glyph.getMembers()) {
                    section.clearViews();
                    tLag.addVertex(section); // Trick!
                    section.setGraph(tLag);
                    section.complete();
                }

                int viewIndex = tLag.viewIndexOf(view);
                view.colorizeGlyph(viewIndex, glyph);
            }

            return glyph;
        }

        //----------//
        // setIndex //
        //----------//
        /**
         * Only method allowed to designate a glyph
         *
         * @param index index of new current glyph
         * @param hint related processing hint
         */
        public void setIndex (int           index,
                              SelectionHint hint)
        {
            Glyph glyph = null;

            if (index >= 0) {
                String gName = names.get(index);
                nameField.setText(gName);

                // Special case for icon : if we point to an icon, we have to
                // get rid of all other icons (standard glyphs can be kept)
                if (repository.isIcon(gName)) {
                    repository.unloadIconsFrom(names);
                }

                // Load the desired glyph if needed
                glyph = getGlyph(gName);

                if (glyph == null) {
                    return;
                }

                // Extend view model size if needed
                Rectangle box = glyph.getContourBox();
                modelRectangle = modelRectangle.union(box);

                Dimension newSize = modelRectangle.getSize();

                if (!newSize.equals(modelSize)) {
                    modelSize = newSize;
                    view.setModelSize(modelSize);
                }
            } else {
                nameField.setText("");
            }

            tLag.getSelectionService()
                .publish(new GlyphEvent(this, hint, null, glyph));
            nameIndex = index;

            // Enable buttons according to glyph selection
            all.setEnabled(!names.isEmpty());
            prev.setEnabled(index > 0);
            next.setEnabled((index >= 0) && (index < (names.size() - 1)));
        }

        // Just to please the Board interface
        public void onEvent (UserEvent event)
        {
            throw new UnsupportedOperationException("Not supported yet.");
        }

        //--------------//
        // defineLayout //
        //--------------//
        private void defineLayout ()
        {
            CellConstraints cst = new CellConstraints();
            FormLayout      layout = Panel.makeFormLayout(4, 3);
            PanelBuilder    builder = new PanelBuilder(
                layout,
                super.getBody());
            builder.setDefaultDialogBorder();

            int r = 1; // --------------------------------
            builder.add(load, cst.xy(11, r));

            r += 2; // --------------------------------
            builder.add(all, cst.xy(3, r));
            builder.add(prev, cst.xy(7, r));
            builder.add(next, cst.xy(11, r));

            r += 2; // --------------------------------

            JLabel file = new JLabel("File", SwingConstants.RIGHT);
            builder.add(file, cst.xy(1, r));

            nameField.getField()
                     .setHorizontalAlignment(JTextField.LEFT);
            builder.add(nameField.getField(), cst.xyw(3, r, 9));
        }
    }

    //------------//
    // LoadAction //
    //------------//
    private class LoadAction
        extends AbstractAction
    {
        //~ Constructors -------------------------------------------------------

        public LoadAction ()
        {
            super("Load");
        }

        //~ Methods ------------------------------------------------------------

        @Implement(ActionListener.class)
        public void actionPerformed (ActionEvent e)
        {
            // Get a (shrinkable, to allow deletions) list of glyph names
            names = new ArrayList<String>(
                Arrays.asList(verifier.getGlyphNames()));

            // Reset lag & display
            resetBrowser();

            // Set navigator on first glyph, if any
            if (!names.isEmpty()) {
                navigator.setIndex(0, GLYPH_INIT);
            } else {
                if (e != null) {
                    logger.warning("No glyphs selected in Glyph Selector");
                }

                navigator.all.setEnabled(false);
                navigator.prev.setEnabled(false);
                navigator.next.setEnabled(false);
            }
        }
    }
}
