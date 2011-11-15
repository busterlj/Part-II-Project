//----------------------------------------------------------------------------//
//                                                                            //
//                           M e a s u r e N o d e                            //
//                                                                            //
//----------------------------------------------------------------------------//
// <editor-fold defaultstate="collapsed" desc="hdr">                          //
//  Copyright (C) Herve Bitteur 2000-2010. All rights reserved.               //
//  This software is released under the GNU General Public License.           //
//  Goto http://kenai.com/projects/audiveris to report bugs or suggestions.   //
//----------------------------------------------------------------------------//
// </editor-fold>
package omr.score.entity;

import omr.score.visitor.ScoreVisitor;

import omr.util.Navigable;
import omr.util.TreeNode;

import java.util.Comparator;

/**
 * Class <code>MeasureNode</code> is an abstract class that is subclassed for
 * any PartNode with a containing measure. So this class encapsulates a direct
 * link to the enclosing measure.
 *
 * @author Hervé Bitteur
 */
public abstract class MeasureNode
    extends PartNode
{
    //~ Static fields/initializers ---------------------------------------------

    /**
     * Specific comparator to sort collections of MeasureNode instances,
     * according first to staff index, then to abscissa.
     */
    public static final Comparator<TreeNode> staffComparator = new Comparator<TreeNode>() {
        public int compare (TreeNode tn1,
                            TreeNode tn2)
        {
            MeasureNode mn1 = (MeasureNode) tn1;
            MeasureNode mn2 = (MeasureNode) tn2;
            int         deltaStaff = mn1.getStaff()
                                        .getId() - mn2.getStaff()
                                                      .getId();

            if (deltaStaff != 0) {
                // Staves are different
                return deltaStaff;
            } else {
                // Staves are the same, use abscissae to differentiate
                return mn1.getCenter().x - mn2.getCenter().x;
            }
        }
    };


    //~ Instance fields --------------------------------------------------------

    /** Containing measure */
    @Navigable(false)
    private Measure measure;

    //~ Constructors -----------------------------------------------------------

    //-------------//
    // MeasureNode //
    //-------------//
    /**
     * Create a MeasureNode
     *
     * @param container the (direct) container of the node
     */
    public MeasureNode (PartNode container)
    {
        super(container);

        // Set the measure link
        for (TreeNode c = this; c != null; c = c.getParent()) {
            if (c instanceof Measure) {
                measure = (Measure) c;

                break;
            }
        }
    }

    //~ Methods ----------------------------------------------------------------

    //------------------//
    // getContextString //
    //------------------//
    @Override
    public String getContextString ()
    {
        StringBuilder sb = new StringBuilder(super.getContextString());
        sb.append("M")
          .append(getMeasure().getId());

        if (getStaff() != null) {
            sb.append("F")
              .append(getStaff().getId());
        }

        return sb.toString();
    }

    //------------//
    // getMeasure //
    //------------//
    /**
     * Report the containing measure
     *
     * @return the containing measure entity
     */
    public Measure getMeasure ()
    {
        return measure;
    }

    //--------//
    // accept //
    //--------//
    @Override
    public boolean accept (ScoreVisitor visitor)
    {
        return visitor.visit(this);
    }
}
