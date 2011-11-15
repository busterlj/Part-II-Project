//----------------------------------------------------------------------------//
//                                                                            //
//                              P a r t I n f o                               //
//                                                                            //
//----------------------------------------------------------------------------//
// <editor-fold defaultstate="collapsed" desc="hdr">                          //
//  Copyright (C) Herve Bitteur 2000-2010. All rights reserved.               //
//  This software is released under the GNU General Public License.           //
//  Goto http://kenai.com/projects/audiveris to report bugs or suggestions.   //
//----------------------------------------------------------------------------//
// </editor-fold>
package omr.sheet;

import java.util.ArrayList;
import java.util.List;

/**
 * Class <code>PartInfo</code> is the physical gathering of StaffInfo instances
 * in a part
 *
 * @author Hervé Bitteur
 */
public class PartInfo
{
    //~ Instance fields --------------------------------------------------------

    /** Staves in this part */
    private List<StaffInfo> staves = new ArrayList<StaffInfo>();

    //~ Constructors -----------------------------------------------------------

    //----------//
    // PartInfo //
    //----------//
    /** Creates a new instance of PartInfo */
    public PartInfo ()
    {
    }

    //~ Methods ----------------------------------------------------------------

    //-----------//
    // getStaves //
    //-----------//
    public List<StaffInfo> getStaves ()
    {
        return staves;
    }

    //----------//
    // addStaff //
    //----------//
    public void addStaff (StaffInfo staffInfo)
    {
        staves.add(staffInfo);
    }
}
