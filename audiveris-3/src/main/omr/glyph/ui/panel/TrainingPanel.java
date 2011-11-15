//----------------------------------------------------------------------------//
//                                                                            //
//                         T r a i n i n g P a n e l                          //
//                                                                            //
//----------------------------------------------------------------------------//
// <editor-fold defaultstate="collapsed" desc="hdr">                          //
//  Copyright (C) Herve Bitteur 2000-2010. All rights reserved.               //
//  This software is released under the GNU General Public License.           //
//  Goto http://kenai.com/projects/audiveris to report bugs or suggestions.   //
//----------------------------------------------------------------------------//
// </editor-fold>
package omr.glyph.ui.panel;

import omr.glyph.GlyphEvaluator;
import omr.glyph.GlyphNetwork;
import omr.glyph.Shape;
import static omr.glyph.Shape.*;
import omr.glyph.facets.Glyph;
import omr.glyph.ui.GlyphRepository;
import static omr.glyph.ui.panel.GlyphTrainer.Task.Activity.*;

import omr.log.Logger;

import omr.math.NeuralNetwork;

import omr.ui.util.Panel;

import omr.util.Implement;

import com.jgoodies.forms.builder.PanelBuilder;
import com.jgoodies.forms.layout.CellConstraints;
import com.jgoodies.forms.layout.FormLayout;

import org.jdesktop.swingworker.SwingWorker;

import java.awt.event.*;
import java.util.*;

import javax.swing.*;

/**
 * Class <code>TrainingPanel</code> is a panel dedicated to the training of an
 * evaluator. It is used through its subclasses {@link NetworkPanel} and
 * {@link RegressionPanel} to train the neural network evaluator and the linear
 * evaluator respectively. It is a dedicated companion of class
 * {@link GlyphTrainer}.
 *
 * @author Hervé Bitteur
 */
class TrainingPanel
    implements GlyphEvaluator.Monitor, Observer
{
    //~ Static fields/initializers ---------------------------------------------

    /** Usual logger utility */
    private static final Logger logger = Logger.getLogger(TrainingPanel.class);

    //~ Instance fields --------------------------------------------------------

    /** The swing component */
    protected final Panel component;

    /** Current activity (selecting the population, or training the evaluator on
       the selected population */
    protected final GlyphTrainer.Task task;

    /** User action to launch the training */
    protected TrainAction trainAction;

    /** The underlying evaluator to be trained */
    protected GlyphEvaluator evaluator;

    /** User progress bar to visualize the training process */
    protected JProgressBar progressBar = new JProgressBar();

    /** Common JGoodies constraints for this class and its subclass if any */
    protected CellConstraints cst = new CellConstraints();

    /** Common JGoodies builder for this class and its subclass if any */
    protected PanelBuilder builder;

    /** Repository of known glyphs */
    private final GlyphRepository repository = GlyphRepository.getInstance();

    /**
     * Flag to indicate that the whole population of recorded glyphs (and not
     * just the core ones) is to be considered
     */
    private boolean useWhole = true;

    /** Display of cardinality of whole population */
    private JLabel wholeNumber = new JLabel();

    /** Display of cardinality of core population */
    private JLabel coreNumber = new JLabel();

    /** UI panel dealing with repository selection */
    private final SelectionPanel selectionPanel;

    /** The Neural Network evaluator */
    private GlyphNetwork network = GlyphNetwork.getInstance();

    //~ Constructors -----------------------------------------------------------

    //---------------//
    // TrainingPanel //
    //---------------//
    /**
     * Creates a new TrainingPanel object.
     *
     * @param task the current training task
     * @param standardWidth standard width for fields & buttons
     * @param evaluator the underlying evaluator to train
     * @param selectionPanel user panel for glyphs selection
     * @param totalRows total number of display rows, interlines not counted
     */
    public TrainingPanel (GlyphTrainer.Task task,
                          String            standardWidth,
                          GlyphEvaluator    evaluator,
                          SelectionPanel    selectionPanel,
                          int               totalRows)
    {
        this.evaluator = evaluator;
        this.task = task;
        this.selectionPanel = selectionPanel;

        component = new Panel();
        component.setNoInsets();

        FormLayout layout = Panel.makeFormLayout(
            totalRows,
            4,
            "",
            standardWidth,
            standardWidth);

        builder = new PanelBuilder(layout, component);
        builder.setDefaultDialogBorder(); // Useful ?

        defineLayout();
    }

    //~ Methods ----------------------------------------------------------------

    //--------------//
    // getComponent //
    //--------------//
    /**
     * Give access to the encapsulated swing component
     *
     * @return the user panel
     */
    public JComponent getComponent ()
    {
        return component;
    }

    @Implement(NeuralNetwork.Monitor.class)
    public void epochEnded (int    epochIndex,
                            double mse)
    {
    }

    @Implement(GlyphEvaluator.Monitor.class)
    public void glyphProcessed (final Glyph glyph)
    {
    }

    @Implement(NeuralNetwork.Monitor.class)
    public void trainingStarted (final int    epochIndex,
                                 final double mse)
    {
    }

    //--------//
    // update //
    //--------//
    /**
     * Method triggered by new task activity : the train action is enabled only
     * when no activity is going on.
     *
     * @param obs the task object
     * @param unused not used
     */
    @Implement(Observer.class)
    public void update (Observable obs,
                        Object     unused)
    {
        switch (task.getActivity()) {
        case INACTIVE :
            trainAction.setEnabled(true);

            break;

        case SELECTING :
        case TRAINING :
            trainAction.setEnabled(false);

            break;
        }
    }

    //----------//
    // useWhole //
    //----------//
    /**
     * Tell whether the whole glyph base is to be used, or just the core base
     *
     * @return true if whole, false if core
     */
    public boolean useWhole ()
    {
        return useWhole;
    }

    //--------------//
    // defineLayout //
    //--------------//
    /**
     * Define the common part of the layout, each subclass being able to augment
     * this layout from its constructor
     */
    protected void defineLayout ()
    {
        // Buttons to select just the core glyphs, or the whole population
        CoreAction   coreAction = new CoreAction();
        JRadioButton coreButton = new JRadioButton(coreAction);
        WholeAction  wholeAction = new WholeAction();
        JRadioButton wholeButton = new JRadioButton(wholeAction);

        // Group the radio buttons.
        ButtonGroup group = new ButtonGroup();
        group.add(wholeButton);
        wholeButton.setToolTipText("Use the whole glyph base for any action");
        group.add(coreButton);
        coreButton.setToolTipText(
            "Use only the core glyph base for any action");
        wholeButton.setSelected(true);

        // Evaluator Title & Progress Bar
        int    r = 1; // ----------------------------
        String title = evaluator.getName() + " Training";
        builder.addSeparator(title, cst.xyw(1, r, 7));
        builder.add(progressBar, cst.xyw(9, r, 7));

        r += 2; // ----------------------------
        builder.add(wholeButton, cst.xy(3, r));
        builder.add(wholeNumber, cst.xy(5, r));

        r += 2; // ----------------------------
        builder.add(coreButton, cst.xy(3, r));
        builder.add(coreNumber, cst.xy(5, r));

        // Initialize with population cardinalities
        coreAction.actionPerformed(null);
        wholeAction.actionPerformed(null);
    }

    //-----------------//
    // checkPopulation //
    //-----------------//
    private void checkPopulation (List<Glyph> glyphs)
    {
        // Check that all trainable shapes are present in the training
        // population and that only legal shapes are present. If illegal
        // (non trainable) shapes are found, they are removed from the
        // population.
        boolean[] present = new boolean[LAST_PHYSICAL_SHAPE.ordinal() + 1];
        Arrays.fill(present, false);

        for (Iterator<Glyph> it = glyphs.iterator(); it.hasNext();) {
            Glyph glyph = it.next();
            Shape shape = glyph.getShape();

            try {
                if (shape.isTrainable()) {
                    present[shape.ordinal()] = true;
                } else {
                    logger.warning("Removing non trainable shape:" + shape);
                    it.remove();
                }
            } catch (Exception ex) {
                logger.warning("Removing weird shape: " + shape);
                it.remove();
            }
        }

        for (int i = 0; i < present.length; i++) {
            if (!present[i]) {
                logger.warning("Missing shape: " + Shape.values()[i]);
            }
        }
    }

    //~ Inner Classes ----------------------------------------------------------

    //------------//
    // DumpAction //
    //------------//
    protected class DumpAction
        extends AbstractAction
    {
        //~ Constructors -------------------------------------------------------

        public DumpAction ()
        {
            super("Dump");
        }

        //~ Methods ------------------------------------------------------------

        @Implement(ActionListener.class)
        public void actionPerformed (ActionEvent e)
        {
            evaluator.dump();
        }
    }

    //-------------//
    // TrainAction //
    //-------------//
    protected class TrainAction
        extends AbstractAction
    {
        //~ Instance fields ----------------------------------------------------

        // Specific training starting mode
        protected GlyphEvaluator.StartingMode mode = GlyphEvaluator.StartingMode.SCRATCH;
        protected boolean                     confirmationRequired = true;

        //~ Constructors -------------------------------------------------------

        public TrainAction (String title)
        {
            super(title);
        }

        //~ Methods ------------------------------------------------------------

        @Implement(ActionListener.class)
        public void actionPerformed (ActionEvent e)
        {
            // Ask user confirmation
            if (confirmationRequired) {
                int answer = JOptionPane.showConfirmDialog(
                    component,
                    "Do you really want to retrain from scratch?" +
                    "\nMake sure you have a backup of files:" +
                    "\n- config/neural-network.xml" +
                    "\n- config/linear-evaluator.xml");

                if (answer != JOptionPane.YES_OPTION) {
                    return;
                }
            }

            class Worker
                extends Thread
            {
                @Override
                public void run ()
                {
                    train();
                }
            }

            Worker worker = new Worker();
            worker.setPriority(Thread.MIN_PRIORITY);
            worker.start();
        }

        //-------//
        // train //
        //-------//
        public void train ()
        {
            task.setActivity(TRAINING);

            Collection<String> gNames = selectionPanel.getBase(useWhole);
            progressBar.setValue(0);
            progressBar.setMaximum(network.getListEpochs());

            List<Glyph> glyphs = new ArrayList<Glyph>();

            for (String gName : gNames) {
                Glyph glyph = repository.getGlyph(gName, selectionPanel);

                if (glyph != null) {
                    glyphs.add(glyph);
                } else {
                    logger.warning("Cannot get glyph " + gName);
                }
            }

            // Check that all trainable shapes (and only those ones) are
            // present in the training population
            checkPopulation(glyphs);

            evaluator.train(glyphs, TrainingPanel.this, mode);

            task.setActivity(INACTIVE);
        }
    }

    //------------//
    // CoreAction //
    //------------//
    private class CoreAction
        extends AbstractAction
    {
        //~ Instance fields ----------------------------------------------------

        final SwingWorker<Integer, Object> worker = new SwingWorker<Integer, Object>() {
            @Override
            public void done ()
            {
                try {
                    coreNumber.setText("" + get());
                } catch (Exception ex) {
                    logger.warning("Error while loading core base", ex);
                }
            }

            @Override
            protected Integer doInBackground ()
            {
                return selectionPanel.getBase(false)
                                     .size();
            }
        };


        //~ Constructors -------------------------------------------------------

        public CoreAction ()
        {
            super("Core");
        }

        //~ Methods ------------------------------------------------------------

        @Implement(ActionListener.class)
        public void actionPerformed (ActionEvent e)
        {
            useWhole = false;
            worker.execute();
        }
    }

    //-------------//
    // WholeAction //
    //-------------//
    private class WholeAction
        extends AbstractAction
    {
        //~ Instance fields ----------------------------------------------------

        final SwingWorker<Integer, Object> worker = new SwingWorker<Integer, Object>() {
            @Override
            public void done ()
            {
                try {
                    wholeNumber.setText("" + get());
                } catch (Exception ex) {
                    logger.warning("Error while loading whole base", ex);
                }
            }

            @Override
            protected Integer doInBackground ()
            {
                return selectionPanel.getBase(true)
                                     .size();
            }
        };


        //~ Constructors -------------------------------------------------------

        public WholeAction ()
        {
            super("Whole");
        }

        //~ Methods ------------------------------------------------------------

        @Implement(ActionListener.class)
        public void actionPerformed (ActionEvent e)
        {
            useWhole = true;
            worker.execute();
        }
    }
}
