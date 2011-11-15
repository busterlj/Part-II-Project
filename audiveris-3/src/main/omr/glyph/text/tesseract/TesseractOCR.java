//----------------------------------------------------------------------------//
//                                                                            //
//                          T e s s e r a c t O C R                           //
//                                                                            //
//----------------------------------------------------------------------------//
// <editor-fold defaultstate="collapsed" desc="hdr">                          //
//  Copyright (C) Herve Bitteur 2000-2010. All rights reserved.               //
//  This software is released under the GNU General Public License.           //
//  Goto http://kenai.com/projects/audiveris to report bugs or suggestions.   //
//----------------------------------------------------------------------------//
// </editor-fold>
package omr.glyph.text.tesseract;

import omr.WellKnowns;

import omr.glyph.text.*;

import omr.log.Logger;

import omr.score.common.PixelRectangle;

import omr.util.Implement;
import omr.util.OmrExecutors;
import omr.util.ClassUtil;

import net.gencsoy.tesjeract.EANYCodeChar;
import net.gencsoy.tesjeract.Tesjeract;

import java.awt.image.BufferedImage;
import java.io.*;
import java.nio.ByteBuffer;
import java.util.*;
import java.util.concurrent.Callable;

import javax.imageio.ImageIO;
import javax.imageio.ImageWriter;
import javax.imageio.stream.ImageOutputStream;

/**
 * Class <code>TesseractOCR</code> is an OCR service built on the Google
 * Tesseract engine.
 *
 * <p>It relies on the <b>tessdll.dll</b> library, accessed through the
 * <b>tesjeract</b> Java interface.</p>
 *
 * @author Hervé Bitteur
 */
public class TesseractOCR
    implements OCR
{
    //~ Static fields/initializers ---------------------------------------------

    /** Usual logger utility */
    private static final Logger logger = Logger.getLogger(TesseractOCR.class);

    /** Singleton */
    private static final OCR INSTANCE = new TesseractOCR();

    /** Folder where all OCR material is available */
    public static final File ocrHome = WellKnowns.OCR_FOLDER;

    //~ Instance fields --------------------------------------------------------

    /** The OS-dependent chars retriever */
    private boolean tesjeractLoaded = false;

    /** Permanent flag to avoid endless error messages */
    private boolean userWarned = false;

    //~ Constructors -----------------------------------------------------------

    /**
     * Creates the TesseractOCR singleton.
     */
    private TesseractOCR ()
    {
    }

    //~ Methods ----------------------------------------------------------------

    //-------------//
    // getInstance //
    //-------------//
    /**
     * Report the singleton
     * @return the TesseractOCR instance
     */
    public static OCR getInstance ()
    {
        return INSTANCE;
    }

    //-----------------------//
    // getSupportedLanguages //
    //-----------------------//
    /**
     * {@inheritDoc}
     */
    @Implement(OCR.class)
    public Set<String> getSupportedLanguages ()
    {
        if (retrieverInstalled()) {
            try {
                return Tesjeract.getLanguages();
            } catch (Exception ex) {
                logger.warning("Error in loading languages", ex);
            }
        }

        return new HashSet<String>();
    }

    //-----------//
    // recognize //
    //-----------//
    /**
     * {@inheritDoc}
     */
    @Implement(OCR.class)
    public synchronized List<OcrLine> recognize (BufferedImage image,
                                                 final String  languageCode,
                                                 final String  label)
    {
        // Make sure we have a retriever
        if (!retrieverInstalled()) {
            return null;
        }

        try {
            final ByteBuffer        buf = imageToTiffBuffer(image);

            // Delegate the processing to the specific OCR thread
            Callable<List<OcrLine>> task = new Callable<List<OcrLine>>() {
                public List<OcrLine> call ()
                    throws Exception
                {
                    synchronized (this) {
                        EANYCodeChar[] chars = new Tesjeract(languageCode).recognizeAllWords(buf);
                        return getLines(chars, label);
                    }
                }
            };

            // Launch task and wait for its result ...
            return OmrExecutors.getOcrExecutor()
                               .submit(task)
                               .get();
        } catch (Throwable ex) {
            logger.warning("Error in OCR recognize", ex);

            return null;
        }
    }

    //----------//
    // getLines //
    //----------//
    private List<OcrLine> getLines (EANYCodeChar[] chars,
                                    String         label)
    {
        if (logger.isFineEnabled()) {
            dumpChars(chars, label);
        }

        List<OcrLine> lines = new ArrayList<OcrLine>();
        List<OcrChar> lineChars = new ArrayList<OcrChar>();
        int           lastPointSize = -1;

        try {
            for (int index = 0; index < chars.length; index++) {
                EANYCodeChar ch = chars[index];

                // Compute the number of bytes for this UTF8 sequence
                int byteCount = utf8ByteCount(ch.char_code);
                lineChars.add(buildCharDesc(chars, index, byteCount));

                // Let's move to the very last byte of the sequence
                // To use its formatting data
                index += (byteCount - 1);
                ch = chars[index];
                lastPointSize = ch.point_size;

                // End of line?
                if (isNewLine(ch)) {
                    lines.add(new OcrLine(lastPointSize, lineChars, null));
                    lineChars.clear();
                }
            }

            // Debugging: we've found nothing
            if (lines.isEmpty()) {
                dumpChars(chars, label);
            }

            // TODO: (is this useful?) Just in case we've missed the end
            if (!lineChars.isEmpty()) {
                lines.add(new OcrLine(lastPointSize, lineChars, null));
            }

            return lines;
        } catch (Exception ex) {
            logger.warning("Error decoding tesseract output", ex);
            dumpChars(chars, label);

            return null;
        }
    }

    //-----------//
    // isNewLine //
    //-----------//
    /**
     * Report whether this char is the last one of a line
     * @param ch the char descriptor
     * @return true if end of line
     */
    private static boolean isNewLine (EANYCodeChar ch)
    {
        return ((ch.formatting & 0x40) != 0) // newLine
                ||((ch.formatting & 0x80) != 0); // newPara
    }

    //---------------//
    // buildCharDesc //
    //---------------//
    /**
     * Extract and translate the character value out of the UTF8 sequence, while
     * fixing extension character if any
     * @param chars the global char sequence returned by Tesseract
     * @param index the starting index in the chars sequence
     * @param byteCount the number of bytes of the UTF8 sequence
     * @return the proper string value
     * @throws java.io.UnsupportedEncodingException if UTF8 sequence is wrong
     */
    private OcrChar buildCharDesc (EANYCodeChar[] chars,
                                   int            index,
                                   int            byteCount)
        throws UnsupportedEncodingException
    {
        EANYCodeChar   ch = chars[index];

        // Copy char box information (with slight corrections)
        PixelRectangle box = new PixelRectangle(
            ch.left,
            ch.top + 1, // Correction
            ch.right - ch.left,
            ch.bottom - ch.top);

        // Get correct string value
        String str;

        // Check for extension character badly recognized, using aspect
        double aspect = (double) box.width / (double) box.height;

        if (aspect >= TextInfo.getMinExtensionAspect()) {
            if (logger.isFineEnabled()) {
                logger.fine("Suspecting an Extension character");
            }

            str = TextInfo.EXTENSION_STRING;
        } else {
            byte[] bytes = new byte[1000];

            for (int i = 0; i < byteCount; i++) {
                bytes[i] = (byte) chars[index + i].char_code;
            }

            str = new String(Arrays.copyOf(bytes, byteCount), "UTF8");
        }

        return new OcrChar(str, box, ch.point_size, ch.blanks);
    }

    //-----------//
    // dumpChars //
    //-----------//
    /**
     * Dump the raw char descriptions as read from Tesseract
     * @param chars the sequence of raw char descriptions
     * @param the optional label
     */
    private static void dumpChars (EANYCodeChar[] chars,
                                   String         label)
    {
        System.out.println(
            "-- " + ((label != null) ? label : "") + " Raw Tesseract output:");
        System.out.println(
            "char     code  left right   top   bot  font  conf  size blanks   format");

        for (EANYCodeChar ch : chars) {
            System.out.println(
                String.format(
                    "%3s %5d=%2Xh %5d %5d %5d %5d %5d %5d %5d %5d %5d=%2Xh",
                    String.copyValueOf(Character.toChars(ch.char_code)),
                    ch.char_code,
                    ch.char_code,
                    ch.left,
                    ch.right,
                    ch.top,
                    ch.bottom,
                    ch.font_index,
                    ch.confidence,
                    ch.point_size,
                    ch.blanks,
                    ch.formatting,
                    ch.formatting));
        }
    }

    //--------------------//
    // retrieverInstalled //
    //--------------------//
    /**
     * Make sure the OCR retriever is properly installed
     * @return true if OK
     */
    private boolean retrieverInstalled ()
    {
        if (!tesjeractLoaded) {
            if (userWarned) {
                return false;
            }

            try {
                try {
                    ClassUtil.loadLibrary("tesjeract");
                } catch (UnsatisfiedLinkError ex) {
                    String arch = System.getProperty("os.arch");

                    if (arch.equals("amd64")) {
                        arch = "x86_64";
                    } else if (arch.endsWith("86")) {
                        arch = "x86";
                    }

                    if (WellKnowns.WINDOWS) {
                        ClassUtil.load(new File(TesseractOCR.ocrHome, "tessdll.dll"));
                        ClassUtil.load(new File(TesseractOCR.ocrHome, "tesjeract.dll"));
                    } else if (WellKnowns.LINUX) {
                        ClassUtil.load(new File(TesseractOCR.ocrHome, "libtesjeract-linux-" + arch + ".so"));
                    } else if (WellKnowns.MAC_OS_X) {
                        ClassUtil.load(new File(TesseractOCR.ocrHome, "libtesjeract-macosx-" + arch + ".so"));
                    }
                }

                Tesjeract.setTessdataFallback(TesseractOCR.ocrHome + "/");
            } catch (Throwable ex) {
                userWarned = true;
                logger.warning("Could not load Tesjeract");
                return false;
            }
        }

        tesjeractLoaded = true;
        return true;
    }

    //-------------------//
    // imageToTiffBuffer //
    //-------------------//
    /**
     * Convert the given image into TIFF format and return as a
     * ByteBuffer for passing directly to Tesseract
     * @param image the input image
     * @return the image in TIFF format
     */
    private ByteBuffer imageToTiffBuffer (BufferedImage image)
        throws IOException
    {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ImageOutputStream     ios = ImageIO.createImageOutputStream(baos);

        // Take the first suitable TIFF writer
        ImageWriter writer = ImageIO.getImageWritersByFormatName("tiff")
                                    .next();
        writer.setOutput(ios);
        writer.write(image);
        ios.close();

        // allocate() doesn't work
        ByteBuffer buf = ByteBuffer.allocateDirect(baos.size());
        buf.put(baos.toByteArray());

        return buf;
    }

    //---------------//
    // utf8ByteCount //
    //---------------//
    /**
     * Return the number of bytes of this UTF8 sequence
     * @param code the char code of the first byte of the sequence
     * @return the number of bytes for this sequence (or 0 if the byte is not
     * a sequence starting byte)
     */
    private static int utf8ByteCount (int code)
    {
        // Unicode          Byte1    Byte2    Byte3    Byte4
        // -------          -----    -----    -----    -----
        // U+0000-U+007F    0xxxxxxx
        // U+0080-U+07FF    110yyyxx 10xxxxxx
        // U+0800-U+FFFF    1110yyyy 10yyyyxx 10xxxxxx
        // U+10000-U+10FFFF 11110zzz 10zzyyyy 10yyyyxx 10xxxxxx
        if ((code & 0x80) == 0x00) {
            return 1;
        }

        if ((code & 0xE0) == 0xC0) {
            return 2;
        }

        if ((code & 0xF0) == 0xE0) {
            return 3;
        }

        if ((code & 0xF8) == 0xF0) {
            return 4;
        }

        // This is not a legal sequence start
        return 0;
    }
}
