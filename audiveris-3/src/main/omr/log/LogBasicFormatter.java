//----------------------------------------------------------------------------//
//                                                                            //
//                     L o g B a s i c F o r m a t t e r                      //
//                                                                            //
//----------------------------------------------------------------------------//
// <editor-fold defaultstate="collapsed" desc="hdr">                          //
//  Copyright (C) Herve Bitteur 2000-2010. All rights reserved.               //
//  This software is released under the GNU General Public License.           //
//  Goto http://kenai.com/projects/audiveris to report bugs or suggestions.   //
//----------------------------------------------------------------------------//
// </editor-fold>
package omr.log;

import omr.util.*;

import java.io.*;
import java.text.*;
import java.util.Date;
import java.util.logging.*;

/**
 * Class <code>LogBasicFormatter</code> formats a log record.
 *
 * @author Hervé Bitteur
 */
public class LogBasicFormatter
    extends Formatter
{
    //~ Static fields/initializers ---------------------------------------------

    /** Line separator string.  This is the value of the line.separator */
    private static String lineSeparator = "\n";

    /** Standard format */
    private static final String format = "{0,time}";

    /** Should we print time in log */
    private static final boolean PRINT_TIME = false;

    /** Should we print the calling frame in log */
    private static final boolean PRINT_FRAME = true;

    /** Classes to skip when retrieving the actual caller of the log */
    private static final Class[] logClasses = new Class[] {
                                                  java.util.logging.Logger.class,
                                                  omr.log.Logger.class
                                              };

    //~ Instance fields --------------------------------------------------------

    private Date          dat = new Date();
    private MessageFormat formatter;
    private Object[]      args = new Object[1];

    //~ Methods ----------------------------------------------------------------

    /**
     * Format the given LogRecord.
     *
     * @param record the log record to be formatted.
     *
     * @return a formatted log record
     */
    public synchronized String format (LogRecord record)
    {
        StringBuilder sb = new StringBuilder(256);

        // First line (if any)
        if (PRINT_TIME) {
            dat.setTime(record.getMillis());
            args[0] = dat;

            StringBuffer text = new StringBuffer();

            if (formatter == null) {
                formatter = new MessageFormat(format);
            }

            formatter.format(args, text, null);
            sb.append(text);
        }

        if (PRINT_FRAME) {
            StackTraceElement frame = ClassUtil.getCallingFrame(logClasses);

            if (frame != null) {
                sb.append(" ")
                  .append(frame);
            }
        }

        if (sb.length() > 0) {
            //sb.append(lineSeparator);
            sb.append(" -- ");
        }

        // Second part
        String message = formatMessage(record);

        //sb.append(record.getLevel().getLocalizedName());
        sb.append(record.getLevel().getName());
        sb.append(": ");
        sb.append(message);
        sb.append(lineSeparator);

        if (record.getThrown() != null) {
            StringWriter sw = new StringWriter();
            PrintWriter  pw = new PrintWriter(sw);
            record.getThrown()
                  .printStackTrace(pw);
            pw.close();
            sb.append(sw.toString());

            //            Throwable cause = record.getThrown()
            //                                    .getCause();
            //
            //            if (cause != null) {
            //                sb.append(" cause:")
            //                  .append(cause.getMessage());
            //            }
        }

        return sb.toString();
    }
}
