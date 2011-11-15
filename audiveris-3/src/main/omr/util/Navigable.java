//----------------------------------------------------------------------------//
//                                                                            //
//                             N a v i g a b l e                              //
//                                                                            //
//----------------------------------------------------------------------------//
// <editor-fold defaultstate="collapsed" desc="hdr">                          //
//  Copyright (C) Herve Bitteur 2000-2010. All rights reserved.               //
//  This software is released under the GNU General Public License.           //
//  Goto http://kenai.com/projects/audiveris to report bugs or suggestions.   //
//----------------------------------------------------------------------------//
// </editor-fold>
package omr.util;

import java.lang.annotation.*;

/**
 * Annotation <code>Navigable</code> is used to drive the browsing done by
 * ScoreTree.
 */
@Documented
@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.FIELD)
public @interface Navigable {
    /**
     * @return whether the annotated field can be navigated
     */
    boolean value() default true;
}
