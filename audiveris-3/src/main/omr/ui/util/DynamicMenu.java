//----------------------------------------------------------------------------//
//                                                                            //
//                           D y n a m i c M e n u                            //
//                                                                            //
//----------------------------------------------------------------------------//
// <editor-fold defaultstate="collapsed" desc="hdr">                          //
//  Copyright (C) Herve Bitteur 2000-2010. All rights reserved.               //
//  This software is released under the GNU General Public License.           //
//  Goto http://kenai.com/projects/audiveris to report bugs or suggestions.   //
//----------------------------------------------------------------------------//
// </editor-fold>
package omr.ui.util;

import javax.swing.*;
import javax.swing.event.MenuEvent;
import javax.swing.event.MenuListener;

/**
 * Class <code>DynamicMenu</code> simplifies the definition of a menu,
 * whose content needs to be updated on-the-fly when the menu is being
 * selected.
 *
 * @author Hervé Bitteur
 */
public abstract class DynamicMenu
{
    //~ Instance fields --------------------------------------------------------

    /** The concrete UI menu */
    private final JMenu menu;
    private MenuListener listener = new MenuListener() {
        public void menuCanceled (MenuEvent e)
        {
        }

        public void menuDeselected (MenuEvent e)
        {
        }

        public void menuSelected (MenuEvent e)
        {
            // Clean up the whole menu
            menu.removeAll();

            // Rebuild the whole list of menu items on the fly
            buildItems();
        }
    };


    //~ Constructors -----------------------------------------------------------

    //-------------//
    // DynamicMenu //
    //-------------//
    /**
     * Create the dynamic menu
     *
     * @param menuLabel the label to be used for the menu
     */
    public DynamicMenu (String menuLabel)
    {
        menu = new JMenu(menuLabel);

        // Listener to menu selection, to modify content on-the-fly
        menu.addMenuListener(listener);
    }

    /**
     * Creates a new DynamicMenu object.
     *
     * @param action DOCUMENT ME!
     */
    public DynamicMenu (Action action)
    {
        menu = new JMenu(action);

        // Listener to menu selection, to modify content on-the-fly
        menu.addMenuListener(listener);
    }

    //~ Methods ----------------------------------------------------------------

    //---------//
    // getMenu //
    //---------//
    /**
     * Report the concrete menu
     * @return the usable menu
     */
    public JMenu getMenu ()
    {
        return menu;
    }

    //------------//
    // buildItems //
    //------------//
    /**
     * This is the method that is called whenever the menu is selected, so
     * this is the method which must be implemented in a subclass.
     */
    protected abstract void buildItems ();
}
