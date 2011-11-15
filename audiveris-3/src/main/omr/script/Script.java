//----------------------------------------------------------------------------//
//                                                                            //
//                                S c r i p t                                 //
//                                                                            //
//----------------------------------------------------------------------------//
// <editor-fold defaultstate="collapsed" desc="hdr">                          //
//  Copyright (C) Herve Bitteur 2000-2010. All rights reserved.               //
//  This software is released under the GNU General Public License.           //
//  Goto http://kenai.com/projects/audiveris to report bugs or suggestions.   //
//----------------------------------------------------------------------------//
// </editor-fold>
package omr.script;

import omr.log.Logger;

import omr.sheet.Sheet;

import omr.step.ProcessingCancellationException;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import javax.xml.bind.annotation.*;

/**
 * Class {@code Script} handles a complete script applied to a sheet.
 *
 * <p>A script is a sequence of {@link ScriptTask} instances tasks that are
 * recorded as the user interacts with the sheet data.
 *
 * <p>A script can be stored and reloaded/replayed.
 *
 * @author Hervé Bitteur
 */
@XmlAccessorType(XmlAccessType.NONE)
@XmlRootElement(name = "script")
public class Script
{
    //~ Static fields/initializers ---------------------------------------------

    /** Usual logger utility */
    private static final Logger logger = Logger.getLogger(Script.class);

    //~ Instance fields --------------------------------------------------------

    /** Sheet to which the script is applied */
    private Sheet sheet;

    /** Full path to the Sheet file name */
    @XmlAttribute(name = "sheet")
    private final String sheetPath;

    /** Sequence of tasks that compose the script */
    @XmlElements({@XmlElement(name = "assign", type = AssignTask.class)
        , @XmlElement(name = "barline", type = BarlineTask.class)
        , @XmlElement(name = "boundary", type = BoundaryTask.class)
        , @XmlElement(name = "delete", type = DeleteTask.class)
        , @XmlElement(name = "export", type = ExportTask.class)
        , @XmlElement(name = "insert", type = InsertTask.class)
        , @XmlElement(name = "midi", type = MidiWriteTask.class)
        , @XmlElement(name = "parameters", type = ParametersTask.class)
        , @XmlElement(name = "play", type = PlayTask.class)
        , @XmlElement(name = "rational", type = RationalTask.class)
        , @XmlElement(name = "segment", type = SegmentTask.class)
        , @XmlElement(name = "slur", type = SlurTask.class)
        , @XmlElement(name = "step", type = StepTask.class)
        , @XmlElement(name = "text", type = TextTask.class)
    })
    private final List<ScriptTask> tasks = new ArrayList<ScriptTask>();

    /** Flag a script that needs to be stored */
    private boolean modified;

    //~ Constructors -----------------------------------------------------------

    //--------//
    // Script //
    //--------//
    /**
     * Create a script
     *
     * @param sheet the related sheet
     */
    public Script (Sheet sheet)
    {
        this.sheet = sheet;
        sheetPath = sheet.getPath();
    }

    //--------//
    // Script //
    //--------//
    /** No-arg constructor for JAXB */
    private Script ()
    {
        sheetPath = null;
    }

    //~ Methods ----------------------------------------------------------------

    //------------//
    // isModified //
    //------------//
    /**
     * Has the script been modified (wrt its backup on disk)?
     * @return the modified
     */
    public boolean isModified ()
    {
        return modified;
    }

    //----------//
    // getSheet //
    //----------//
    /**
     * Report the sheet this script is linked to
     *
     * @return the sheet concerned
     */
    public Sheet getSheet ()
    {
        return sheet;
    }

    //---------//
    // addTask //
    //---------//
    /**
     * Add a task to the script
     *
     * @param task the task to add at the end of the current sequence
     */
    public void addTask (ScriptTask task)
    {
        tasks.add(task);
        setModified(true);

        if (logger.isFineEnabled()) {
            logger.fine("Script: added " + task);
        }
    }

    //------//
    // dump //
    //------//
    /**
     * Meant for debug
     */
    public void dump ()
    {
        logger.info(toString());

        for (ScriptTask task : tasks) {
            logger.info(task.toString());
        }
    }

    //-----//
    // run //
    //-----//
    /**
     * This methods runs sequentially and synchronously the various tasks of the
     * script. It is up to the caller to run this method in a separate thread
     * if so desired.
     */
    public void run ()
    {
        if (logger.isFineEnabled()) {
            logger.fine(
                "Running " + this +
                ((sheet != null) ? (" on sheet " + sheet.getRadix()) : ""));
        }

        // Make sheet concrete
        if (sheet == null) {
            if (sheetPath == null) {
                logger.warning("No sheet defined in script");

                return;
            }

            sheet = new Sheet(new File(sheetPath));
        }

        // Run the tasks in sequence
        try {
            for (ScriptTask task : tasks) {
                if (logger.isFineEnabled()) {
                    logger.fine(
                        "Running " + task + " on sheet " + sheet.getRadix());
                }

                try {
                    // Run the task synchronously (prolog/core/epilog)
                    task.run(sheet);
                } catch (ProcessingCancellationException pce) {
                    throw pce;
                } catch (Exception ex) {
                    logger.warning("Error running " + task, ex);
                    throw new RuntimeException(task.toString());
                }
            }

            if (logger.isFineEnabled()) {
                logger.fine("All tasks run on sheet " + sheet.getRadix());
            }
        } catch (ProcessingCancellationException pce) {
            throw pce;
        } catch (Exception ex) {
            logger.warning("Script aborted", ex);
        } finally {
            // Flag the (active) script as up-to-date
            sheet.getScript()
                 .setModified(false);
        }
    }

    //----------//
    // toString //
    //----------//
    @Override
    public String toString ()
    {
        StringBuilder sb = new StringBuilder();
        sb.append("{Script");

        if (modified) {
            sb.append(" modified");
        }

        if (sheetPath != null) {
            sb.append(" ")
              .append(sheetPath);
        } else if (sheet != null) {
            sb.append(" ")
              .append(sheet.getRadix());
        }

        if (tasks != null) {
            sb.append(" tasks:")
              .append(tasks.size());
        }

        sb.append("}");

        return sb.toString();
    }

    //-------------//
    // setModified //
    //-------------//
    /**
     * Flag the script as modified (wrt disk)
     * @param modified the modified to set
     */
    void setModified (boolean modified)
    {
        this.modified = modified;
    }
}
