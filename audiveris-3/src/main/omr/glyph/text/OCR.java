//----------------------------------------------------------------------------//
//                                                                            //
//                                   O C R                                    //
//                                                                            //
//----------------------------------------------------------------------------//
// <editor-fold defaultstate="collapsed" desc="hdr">                          //
//  Copyright (C) Herve Bitteur 2000-2010. All rights reserved.               //
//  This software is released under the GNU General Public License.           //
//  Goto http://kenai.com/projects/audiveris to report bugs or suggestions.   //
//----------------------------------------------------------------------------//
// </editor-fold>
package omr.glyph.text;

import java.awt.image.BufferedImage;
import java.util.List;
import java.util.Set;

/**
 * Interface <code>OCR</code> defines the interaction with an OCR engine.
 *
 * @author Hervé Bitteur
 */
public interface OCR
{
    //~ Methods ----------------------------------------------------------------

    //-----------------------//
    // getSupportedLanguages //
    //-----------------------//
    /**
     * Report the set of supported language codes
     * @return the set of supported 3-letter codes
     */
    Set<String> getSupportedLanguages ();

    //-----------//
    // recognize //
    //-----------//
    /**
     * Launch the recognition of the provided image, whose language is specified
     *
     * @param image the provided textual image
     * @param languageCode the code of the (dominant?) language of the text,
     * or null if this language is unknown
     * @param label an optional label related to the image, null otherwise.
     * This is meant for debugging the temporary files.
     * @return a list of OcrLine instances
     */
    List<OcrLine> recognize (BufferedImage image,
                             String        languageCode,
                             String        label);
}
