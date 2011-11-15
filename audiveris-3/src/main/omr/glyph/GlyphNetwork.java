//----------------------------------------------------------------------------//
//                                                                            //
//                          G l y p h N e t w o r k                           //
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
import omr.constant.ConstantSet;

import omr.glyph.GlyphEvaluator.StartingMode;
import omr.glyph.facets.Glyph;

import omr.log.Logger;

import omr.math.NeuralNetwork;

import java.io.*;
import java.util.*;

import javax.swing.JOptionPane;
import javax.xml.bind.JAXBException;

/**
 * Class <code>GlyphNetwork</code> encapsulates a neural network dedicated to
 * glyph recognition. It wraps the generic {@link omr.math.NeuralNetwork} with
 * application information, for training, storing, loading and using the neural
 * network.
 *
 * <p>The application neural network data is loaded as follows: <ol>
 * <li>It first tries to find a file in the /config sub-folder of the
 * application, looking for a file named 'neural-network.xml' which contains a
 * custom definition of the network, typically after a training.</li>
 * <li>If not found, it falls back reading the default definition from the
 * application resource, reading the /config/neural-network.xml file provided
 * in the distribution jar file.</ol></p>
 *
 * <p>Similarly, after a training of the neural network, the data is stored as
 * the custom definition in the local file 'config/neural-network.xml', which
 * will be picked first when the application is run again.</p>
 *
 * @author Hervé Bitteur
 */
public class GlyphNetwork
    extends GlyphEvaluator
{
    //~ Static fields/initializers ---------------------------------------------

    /** Specific application parameters */
    private static final Constants constants = new Constants();

    /** Usual logger utility */
    private static final Logger logger = Logger.getLogger(GlyphNetwork.class);

    /** The singleton */
    private static volatile GlyphNetwork INSTANCE;

    /** Neural network backup file name */
    private static final String BACKUP_FILE_NAME = "neural-network.xml";

    //~ Instance fields --------------------------------------------------------

    /** The underlying neural network */
    private NeuralNetwork engine;

    //~ Constructors -----------------------------------------------------------

    //--------------//
    // GlyphNetwork //
    //--------------//
    /**
     * Create an instance of glyph neural network
     */
    private GlyphNetwork ()
    {
        // Unmarshal from backup data
        engine = (NeuralNetwork) unmarshal();

        // Basic check
        if (engine != null) {
            if (engine.getOutputSize() != shapeCount) {
                final String msg = "Neural Network data is obsolete," +
                                   " it must be retrained from scratch";
                logger.warning(msg);
                JOptionPane.showMessageDialog(null, msg);

                engine = null;
            }
        }

        if (engine == null) {
            // Get a brand new one (not trained)
            logger.info("Creating a brand new GlyphNetwork");
            engine = createNetwork();
        }
    }

    //~ Methods ----------------------------------------------------------------

    //-------------//
    // getInstance //
    //-------------//
    /**
     * Report the single instance of GlyphNetwork in the Audiveris application
     *
     * @return the instance
     */
    public static GlyphNetwork getInstance ()
    {
        if (INSTANCE == null) {
            INSTANCE = new GlyphNetwork();
        }

        return INSTANCE;
    }

    //-------------------//
    // getAllEvaluations //
    //-------------------//
    @Override
    public Evaluation[] getAllEvaluations (Glyph glyph)
    {
        // If too small, it's just NOISE
        if (!isBigEnough(glyph)) {
            return noiseEvaluations;
        } else {
            double[]     ins = feedInput(glyph, null);
            double[]     outs = new double[shapeCount];
            Evaluation[] evals = new Evaluation[shapeCount];

            engine.run(ins, null, outs);

            for (int s = 0; s < shapeCount; s++) {
                Shape shape = GlyphChecks.specificCheck(
                    Shape.values()[s],
                    glyph);

                if (shape != null) {
                    evals[s] = new Evaluation(shape, 1d / outs[s]);
                } else {
                    evals[s] = new Evaluation(
                        Shape.values()[s],
                        Double.MAX_VALUE);
                }
            }

            // Order the evals from best to worst
            Arrays.sort(evals, comparator);

            return evals;
        }
    }

    //--------------//
    // setAmplitude //
    //--------------//
    /**
     * Set the amplitude value for initial random values (UNUSED)
     *
     * @param amplitude
     */
    public void setAmplitude (double amplitude)
    {
        constants.amplitude.setValue(amplitude);
    }

    //--------------//
    // getAmplitude //
    //--------------//
    /**
     * Selector for the amplitude value (used in initial random values)
     *
     * @return the amplitude value
     */
    public double getAmplitude ()
    {
        return constants.amplitude.getValue();
    }

    //-----------------//
    // setLearningRate //
    //-----------------//
    /**
     * Dynamically modify the learning rate of the neural network for its
     * training task
     *
     * @param learningRate new learning rate to use
     */
    public void setLearningRate (double learningRate)
    {
        constants.learningRate.setValue(learningRate);
        engine.setLearningRate(learningRate);
    }

    //-----------------//
    // getLearningRate //
    //-----------------//
    /**
     * Selector of the current value for network learning rate
     *
     * @return the current learning rate
     */
    public double getLearningRate ()
    {
        return constants.learningRate.getValue();
    }

    //---------------//
    // setListEpochs //
    //---------------//
    /**
     * Modify the upper limit on the number of epochs (training iterations) for
     * the training process
     *
     * @param listEpochs new value for iteration limit
     */
    public void setListEpochs (int listEpochs)
    {
        constants.listEpochs.setValue(listEpochs);
        engine.setEpochs(listEpochs);
    }

    //---------------//
    // getListEpochs //
    //---------------//
    /**
     * Selector on the maximum numner of training iterations
     *
     * @return the upper limit on iteration counter
     */
    public int getListEpochs ()
    {
        return constants.listEpochs.getValue();
    }

    //-------------//
    // setMaxError //
    //-------------//
    /**
     * Modify the error threshold to potentially stop the training process
     *
     * @param maxError the new threshold value to use
     */
    public void setMaxError (double maxError)
    {
        constants.maxError.setValue(maxError);
        engine.setMaxError(maxError);
    }

    //-------------//
    // getMaxError //
    //-------------//
    /**
     * Report the error threshold to potentially stop the training process
     *
     * @return the threshold currently in use
     */
    public double getMaxError ()
    {
        return constants.maxError.getValue();
    }

    //-------------//
    // setMomentum //
    //-------------//
    /**
     * Modify the value for momentum used from learning epoch to the other
     *
     * @param momentum the new momentum value to be used
     */
    public void setMomentum (double momentum)
    {
        constants.momentum.setValue(momentum);
        engine.setMomentum(momentum);
    }

    //-------------//
    // getMomentum //
    //-------------//
    /**
     * Report the momentum training value currently in use
     *
     * @return the momentum in use
     */
    public double getMomentum ()
    {
        return constants.momentum.getValue();
    }

    //---------//
    // getName //
    //---------//
    /**
     * Report a name for this network
     *
     * @return a simple name
     */
    @Override
    public String getName ()
    {
        return "Neural";
    }

    //------------//
    // getNetwork //
    //------------//
    /**
     * Selector to the encapsulated Neural Network
     *
     * @return the neural network
     */
    public NeuralNetwork getNetwork ()
    {
        return engine;
    }

    //------//
    // dump //
    //------//
    /**
     * Dump the internals of the neural network to the standard output
     */
    @Override
    public void dump ()
    {
        engine.dump();
    }

    //------//
    // stop //
    //------//
    /**
     * Forward "Stop" order to the network being trained
     */
    @Override
    public void stop ()
    {
        engine.stop();
    }

    //-------//
    // train //
    //-------//
    /**
     * Train the network using the provided collection of lists
     *
     * @param glyphs  the provided list of glyphs
     * @param monitor the monitoring entity if any
     * @param mode the starting mode of the trainer (scratch, replay or
     * incremental)
     */
    @SuppressWarnings("unchecked")
    public void train (List<Glyph>  glyphs,
                       Monitor      monitor,
                       StartingMode mode)
    {
        if (glyphs.isEmpty()) {
            logger.warning("No glyph to retrain Network Evaluator");

            return;
        }

        int    quorum = constants.quorum.getValue();

        // Determine cardinality for each shape
        List[] shapeGlyphs = new List[shapeCount];

        for (int i = 0; i < shapeGlyphs.length; i++) {
            shapeGlyphs[i] = new ArrayList<Glyph>();
        }

        for (Glyph glyph : glyphs) {
            shapeGlyphs[glyph.getShape()
                             .ordinal()].add(glyph);
        }

        List<Glyph> newGlyphs = new ArrayList<Glyph>();

        for (List l : shapeGlyphs) {
            int     card = 0;
            boolean first = true;

            if (!l.isEmpty()) {
                while (card < quorum) {
                    for (int i = 0; i < l.size(); i++) {
                        newGlyphs.add((Glyph) l.get(i));
                        card++;

                        if (!first && (card >= quorum)) {
                            break;
                        }
                    }

                    first = false;
                }
            }
        }

        // Shuffle the final collection of glyphs
        Collections.shuffle(newGlyphs);

        // Build the collection of patterns from the glyph data
        double[][] inputs = new double[newGlyphs.size()][];
        double[][] desiredOutputs = new double[newGlyphs.size()][];

        int        ig = 0;

        for (Glyph glyph : newGlyphs) {
            double[] ins = new double[paramCount];
            feedInput(glyph, ins);
            inputs[ig] = ins;

            double[] des = new double[shapeCount];
            Arrays.fill(des, 0);

            des[glyph.getShape()
                     .ordinal()] = 1;
            desiredOutputs[ig] = des;

            ig++;
        }

        // Starting options
        if (mode == StartingMode.SCRATCH) {
            engine = createNetwork();
        }

        // Train on the patterns
        engine.train(inputs, desiredOutputs, monitor);
    }

    //-------------//
    // getFileName //
    //-------------//
    @Override
    protected String getFileName ()
    {
        return BACKUP_FILE_NAME;
    }

    //---------//
    // marshal //
    //---------//
    @Override
    protected void marshal (OutputStream os)
        throws FileNotFoundException, IOException, JAXBException
    {
        engine.marshal(os);
    }

    //-----------//
    // unmarshal //
    //-----------//
    @Override
    protected NeuralNetwork unmarshal (InputStream is)
        throws JAXBException, IOException
    {
        return NeuralNetwork.unmarshal(is);
    }

    //---------------//
    // createNetwork //
    //---------------//
    private NeuralNetwork createNetwork ()
    {
        // Note : We allocate a hidden layer with as many cells as the output
        // layer
        NeuralNetwork nn = new NeuralNetwork(
            paramCount,
            shapeCount,
            shapeCount,
            getAmplitude(),
            getLearningRate(),
            getMomentum(),
            getMaxError(),
            getListEpochs());

        return nn;
    }

    //~ Inner Classes ----------------------------------------------------------

    private static final class Constants
        extends ConstantSet
    {
        //~ Instance fields ----------------------------------------------------

        Constant.Ratio   amplitude = new Constant.Ratio(
            0.5,
            "Initial weight amplitude");
        Constant.Ratio   learningRate = new Constant.Ratio(
            0.2,
            "Learning Rate");
        Constant.Integer listEpochs = new Constant.Integer(
            "Epochs",
            8000,
            "Number of epochs for training on list of glyphs");
        Constant.Integer quorum = new Constant.Integer(
            "Glyphs",
            10,
            "Minimum number of glyphs for each shape");
        Evaluation.Doubt maxError = new Evaluation.Doubt(
            1E-3,
            "Threshold to stop training");
        Constant.Ratio   momentum = new Constant.Ratio(
            0.2,
            "Training momentum");
    }
}
