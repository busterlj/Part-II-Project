//----------------------------------------------------------------------------//
//                                                                            //
//                       G l y p h R e p o s i t o r y                        //
//                                                                            //
//----------------------------------------------------------------------------//
// <editor-fold defaultstate="collapsed" desc="hdr">                          //
//  Copyright (C) Herve Bitteur 2000-2010. All rights reserved.               //
//  This software is released under the GNU General Public License.           //
//  Goto http://kenai.com/projects/audiveris to report bugs or suggestions.   //
//----------------------------------------------------------------------------//
// </editor-fold>
package omr.glyph.ui;

import omr.WellKnowns;

import omr.glyph.GlyphSection;
import omr.glyph.Shape;
import omr.glyph.SymbolGlyph;
import omr.glyph.facets.BasicGlyph;
import omr.glyph.facets.Glyph;
import omr.glyph.facets.GlyphValue;
import omr.glyph.facets.StickValue;

import omr.log.Logger;

import omr.sheet.Sheet;

import omr.ui.symbol.ShapeSymbol;
import omr.ui.symbol.SymbolManager;

import omr.util.BlackList;
import omr.util.FileUtil;

import java.io.*;
import java.util.*;

import javax.xml.bind.*;

/**
 * Class <code>GlyphRepository</code> handles the store of known glyphs, across
 * multiple sheets (and possibly multiple runs).
 *
 * <p> A glyph is known by its full name, whose standard format is
 * <B>sheetName/Shape.id.xml</B>, regardless of the area it is stored (this may
 * be the <I>core</I> area or the global <I>sheets</I> area). It can also be an
 * <I>artificial</I> glyph built from a symbol icon, in that case its full name
 * is the similar formats <B>icons/Shape.xml</B> or <B>icons/Shape.nn.xml</B>
 * where "nn" is a differentiating number.
 *
 * <p> The repository handles a private map of all deserialized glyphs so far,
 * since the deserialization is a rather expensive operation.
 *
 * <p> It handles two bases : the "whole base" (all glyphs) and the "core base"
 * (just the glyphs of the core) which are accessible respectively by {@link
 * #getWholeBase} and {@link #getCoreBase} methods.
 *
 * @author Hervé Bitteur
 */
public class GlyphRepository
{
    //~ Static fields/initializers ---------------------------------------------

    /** Usual logger utility */
    private static final Logger logger = Logger.getLogger(
        GlyphRepository.class);

    /** The single instance of this class */
    private static volatile GlyphRepository INSTANCE;

    /** Extension for training files */
    private static final String FILE_EXTENSION = ".xml";

    /** Name of Structures sub-directory */
    private static final String STRUCTURES_NAME = ".structures";

    /** Specific subdirectory for sheet glyphs */
    private static final File sheetsFolder = new File(
        WellKnowns.TRAIN_FOLDER,
        "sheets");

    /** Specific subdirectory for core glyphs */
    private static final File coreFolder = new File(
        WellKnowns.TRAIN_FOLDER,
        "core");

    /** Specific subdirectory for icons */
    private static final File iconsFolder = WellKnowns.SYMBOLS_FOLDER;

    /** Specific filter for glyph files */
    private static final FileFilter glyphFilter = new FileFilter() {
        public boolean accept (File file)
        {
            return file.isDirectory() ||
                   FileUtil.getExtension(file)
                           .equals(FILE_EXTENSION);
        }
    };

    /** Un/marshalling context for use with JAXB */
    private static volatile JAXBContext jaxbContext;

    /** For comparing shape names */
    public static final Comparator<String> shapeComparator = new Comparator<String>() {
        public int compare (String s1,
                            String s2)
        {
            String n1 = GlyphRepository.shapeNameOf(s1);
            String n2 = GlyphRepository.shapeNameOf(s2);

            return n1.compareTo(n2);
        }
    };


    //~ Instance fields --------------------------------------------------------

    /** Core collection of glyphs */
    private volatile List<String> coreBase;

    /** Whole collection of glyphs */
    private volatile List<String> wholeBase;

    /**
     * Map of all glyphs deserialized so far, using full glyph name as key. Full
     * glyph name format is : sheetName/Shape.id.xml
     */
    private final Map<String, Glyph> glyphsMap = new TreeMap<String, Glyph>();

    /** Inverse map */
    private final Map<Glyph, String> namesMap = new HashMap<Glyph, String>();

    //~ Constructors -----------------------------------------------------------

    /** Private singleton constructor */
    private GlyphRepository ()
    {
    }

    //~ Methods ----------------------------------------------------------------

    //-------------//
    // getInstance //
    //-------------//
    /**
     * Report the single instance of this class, after creating it if needed
     *
     * @return the single instance
     */
    public static GlyphRepository getInstance ()
    {
        if (INSTANCE == null) {
            INSTANCE = new GlyphRepository();
        }

        return INSTANCE;
    }

    //-------------//
    // setCoreBase //
    //-------------//
    /**
     * Define the provided collection as the core training material
     *
     * @param base the provided collection
     */
    public synchronized void setCoreBase (List<String> base)
    {
        coreBase = base;
    }

    //----------//
    // getGlyph //
    //----------//
    /**
     * Return a glyph knowing its full glyph name, which is the name of the
     * corresponding training material. If not already done, the glyph is
     * deserialized from the training file, searching first in the icons area,
     * then the core area, then in the global sheets area.
     *
     * @param gName the full glyph name (format is: sheetName/Shape.id.xml)
     * @param monitor the monitor, if any, to be kept informed of glyph loading
     *
     * @return the glyph instance if found, null otherwise
     */
    public Glyph getGlyph (String  gName,
                           Monitor monitor)
    {
        // First, try the map of glyphs
        Glyph glyph = glyphsMap.get(gName);

        if (glyph == null) {
            // If failed, actually load the glyph from XML backup file.
            if (isIcon(gName)) {
                glyph = buildIconGlyph(gName);
            } else {
                File file = new File(coreFolder, gName);

                if (!file.exists()) {
                    file = new File(getSheetsFolder(), gName);

                    if (!file.exists()) {
                        logger.warning(
                            "Unable to find file for glyph " + gName);

                        return null;
                    }
                }

                glyph = buildGlyph(gName, file);
            }

            if (glyph != null) {
                glyphsMap.put(gName, glyph);
                namesMap.put(glyph, gName);
            }
        }

        if (monitor != null) {
            monitor.loadedGlyph(gName);
        }

        return glyph;
    }

    //--------------//
    // getGlyphName //
    //--------------//
    public String getGlyphName (Glyph glyph)
    {
        return namesMap.get(glyph);
    }

    //-------------//
    // getGlyphsIn //
    //-------------//
    /**
     * Report the list of glyph files that are contained within a given
     * directory
     *
     * @param dir the containing directory
     * @return the list of glyph files
     */
    public synchronized List<File> getGlyphsIn (File dir)
    {
        File[] files = listLegalFiles(dir);

        if (files != null) {
            return Arrays.asList(files);
        } else {
            logger.warning("Cannot get files list from dir " + dir);

            return new ArrayList<File>();
        }
    }

    //--------//
    // isIcon //
    //--------//
    public boolean isIcon (String gName)
    {
        return isIcon(new File(gName));
    }

    //------------//
    // fileNameOf //
    //------------//
    /**
     * Report the file name w/o extension of a gName.
     *
     * @param gName glyph name, using format "folder/name.number.xml" or "folder/name.xml"
     * @return the 'name' or 'name.number' part of the format
     */
    public static String fileNameOf (String gName)
    {
        int    slash = gName.indexOf("/");
        String nameWithExt = gName.substring(slash + 1);

        int    lastDot = nameWithExt.lastIndexOf(".");

        if (lastDot != -1) {
            return nameWithExt.substring(0, lastDot);
        } else {
            return nameWithExt;
        }
    }

    //-------------//
    // getCoreBase //
    //-------------//
    /**
     * Return the names of the core collection of glyphs
     *
     * @return the core collection of recorded glyphs
     */
    public List<String> getCoreBase (Monitor monitor)
    {
        if (coreBase == null) {
            synchronized (this) {
                if (coreBase == null) {
                    coreBase = loadCoreBase(monitor);
                }
            }
        }

        return coreBase;
    }

    //---------------------//
    // getSheetDirectories //
    //---------------------//
    /**
     * Report the list of all sheet directories found in the training material
     *
     * @return the list of sheet directories
     */
    public synchronized List<File> getSheetDirectories ()
    {
        // One directory per sheet
        List<File> dirs = new ArrayList<File>();
        File[]     files = listLegalFiles(getSheetsFolder());

        for (File file : files) {
            if (file.isDirectory()) {
                dirs.add(file);
            }
        }

        return dirs;
    }

    //-----------------//
    // getSheetsFolder //
    //-----------------//
    /**
     * Report the folder where all sheet glyphs are stored
     *
     * @return the directory of all sheets material
     */
    public File getSheetsFolder ()
    {
        return sheetsFolder;
    }

    //--------------//
    // getWholeBase //
    //--------------//
    /**
     * Return the names of the whole collection of glyphs
     *
     * @return the whole collection of recorded glyphs
     */
    public List<String> getWholeBase (Monitor monitor)
    {
        if (wholeBase == null) {
            synchronized (this) {
                if (wholeBase == null) {
                    wholeBase = loadWholeBase(monitor);
                }
            }
        }

        return wholeBase;
    }

    //-------------------//
    // recordSheetGlyphs //
    //-------------------//
    /**
     * Store all known glyphs of the provided sheet as separate XML files, so
     * that they can be later re-loaded to train an evaluator. We store glyph
     * for which Shape is not null, and different from NOISE and STEM (CLUTTER
     * is thus stored as well).
     *
     * <p>STRUCTURE shapes are stored in a parallel sub-directory so that they
     * don't get erased by shpes of their leaves.
     *
     * @param sheet the sheet whose glyphs are to be stored
     * @param emptyStructures flag to specify if the Structure directory must be
     * emptied beforehand
     */
    public void recordSheetGlyphs (Sheet   sheet,
                                   boolean emptyStructures)
    {
        // Prepare target directory
        File sheetDir = new File(getSheetsFolder(), sheet.getRadix());

        // Make sure related directory chain exists
        if (sheetDir.mkdirs()) {
            logger.info("Creating directory " + sheetDir);
        } else {
            deleteXmlFiles(sheetDir);
        }

        // Prepare structures directory
        File structuresDir = new File(
            getSheetsFolder(),
            sheet.getRadix() + STRUCTURES_NAME);

        // Make sure related structure subdirectory exists
        if (structuresDir.mkdirs()) {
            logger.info("Creating subdirectory " + structuresDir);
        } else if (emptyStructures) {
            deleteXmlFiles(structuresDir);
        }

        // Now record each relevant glyph
        int glyphNb = 0;
        int structuresNb = 0;

        ///final long startTime = System.currentTimeMillis();
        for (Glyph glyph : sheet.getActiveGlyphs()) {
            if (!glyph.isVirtual() && (glyph.getShape() != null)) {
                Shape shape = glyph.getShape()
                                   .getPhysicalShape();

                if (shape.isTrainable() && (shape != Shape.NOISE)) {
                    {
                        OutputStream os = null;

                        try {
                            if (logger.isFineEnabled()) {
                                logger.fine("Storing " + glyph);
                            }

                            StringBuffer sb = new StringBuffer();
                            sb.append(shape);
                            sb.append(".");
                            sb.append(String.format("%04d", glyph.getId()));
                            sb.append(FILE_EXTENSION);

                            File glyphFile;

                            if (shape != Shape.STRUCTURE) {
                                glyphFile = new File(sheetDir, sb.toString());
                            } else {
                                structuresNb++;
                                glyphFile = new File(
                                    structuresDir,
                                    sb.toString());
                            }

                            os = new FileOutputStream(glyphFile);
                            jaxbMarshal(glyph, os);
                            glyphNb++;
                        } catch (FileNotFoundException ex) {
                            logger.warning(null, ex);
                        } catch (Exception ex) {
                            logger.warning(null, ex);
                        } finally {
                            try {
                                if (os != null) {
                                    os.close();
                                }
                            } catch (IOException ex) {
                                logger.warning(null, ex);
                            }
                        }
                    }
                }
            }
        }

        // Refresh glyph populations
        refreshBases();

        logger.info(
            glyphNb + " glyphs stored from " + sheet.getRadix() +
            ((structuresNb == 0) ? ""
             : (" (including " + structuresNb + " structures)")));
    }

    //--------------//
    // refreshBases //
    //--------------//
    public void refreshBases ()
    {
        wholeBase = null;
        coreBase = null;
    }

    //-------------//
    // shapeNameOf //
    //-------------//
    /**
     * Report the shape name of a gName.
     *
     * @param gName glyph name, using format "folder/name.number.xml" or "folder/name.xml"
     * @return the 'name' part of the format
     */
    public static String shapeNameOf (String gName)
    {
        int    slash = gName.indexOf("/");
        String nameWithExt = gName.substring(slash + 1);

        int    firstDot = nameWithExt.indexOf(".");

        if (firstDot != -1) {
            return nameWithExt.substring(0, firstDot);
        } else {
            return nameWithExt;
        }
    }

    //---------//
    // shapeOf //
    //---------//
    /**
     * Infer the shape of a glyph directly from its full name
     *
     * @param gName the full glyph name
     * @return the shape of the known glyph
     */
    public Shape shapeOf (String gName)
    {
        return shapeOf(new File(gName));
    }

    //---------------//
    // storeCoreBase //
    //---------------//
    /**
     * Store the core training material
     */
    public synchronized void storeCoreBase ()
    {
        if (coreBase == null) {
            logger.warning("Core base is null");

            return;
        }

        // Create the core directory if needed
        File coreDir = coreFolder;
        coreDir.mkdirs();

        // Empty the directory
        FileUtil.deleteAll(coreDir.listFiles());

        // Copy the glyph and icon files into the core directory
        int copyNb = 0;

        for (String gName : coreBase) {
            File source;

            if (isIcon(gName)) {
                source = new File(
                    WellKnowns.SYMBOLS_FOLDER.getParentFile(),
                    gName);
            } else {
                source = new File(getSheetsFolder(), gName);
            }

            File targetDir = new File(
                coreDir,
                source.getParentFile().getName());
            targetDir.mkdirs();

            File target = new File(targetDir, source.getName());

            if (logger.isFineEnabled()) {
                logger.fine("Storing " + target + " as core");
            }

            try {
                FileUtil.copy(source, target);
                copyNb++;
            } catch (IOException ex) {
                logger.warning("Cannot copy " + source + " to " + target);
            }
        }

        logger.info(copyNb + " glyphs copied as core training material");
    }

    //---------------//
    // isIconsFolder //
    //---------------//
    boolean isIconsFolder (String folder)
    {
        return folder.equals(WellKnowns.SYMBOLS_FOLDER.getName());
    }

    //----------//
    // isLoaded //
    //----------//
    boolean isLoaded (String gName)
    {
        return glyphsMap.get(gName) != null;
    }

    //-------------//
    // removeGlyph //
    //-------------//
    /**
     * Remove a glyph from the repository memory (this does not delete the
     * actual glyph file on disk). We also remove it from the various bases
     * which is safer
     *
     * @param gName the full glyph name
     */
    synchronized void removeGlyph (String gName)
    {
        glyphsMap.remove(gName);
        refreshBases();
    }

    //-------------//
    // unloadGlyph //
    //-------------//
    void unloadGlyph (String gName)
    {
        if (glyphsMap.containsKey(gName)) {
            glyphsMap.remove(gName);
        }
    }

    //-----------------//
    // unloadIconsFrom //
    //-----------------//
    void unloadIconsFrom (List<String> names)
    {
        for (String gName : names) {
            if (isIcon(gName)) {
                if (isLoaded(gName)) {
                    Glyph glyph = getGlyph(gName, null);

                    for (GlyphSection section : glyph.getMembers()) {
                        section.clearViews();
                        section.delete();
                    }
                }

                unloadGlyph(gName);
            }
        }
    }

    //--------//
    // isIcon //
    //--------//
    private boolean isIcon (File file)
    {
        String folder = file.getParentFile()
                            .getName();

        return isIconsFolder(folder);
    }

    //----------------//
    // getJaxbContext //
    //----------------//
    private JAXBContext getJaxbContext ()
        throws JAXBException
    {
        // Lazy creation
        if (jaxbContext == null) {
            jaxbContext = JAXBContext.newInstance(
                GlyphValue.class,
                StickValue.class);
        }

        return jaxbContext;
    }

    //------------//
    // buildGlyph //
    //------------//
    private Glyph buildGlyph (String gName,
                              File   file)
    {
        if (logger.isFineEnabled()) {
            logger.fine("Loading glyph " + file);
        }

        Glyph       glyph = null;
        InputStream is = null;

        try {
            is = new FileInputStream(file);
            glyph = jaxbUnmarshal(is);
        } catch (Exception ex) {
            logger.warning("Could not unmarshal file " + file);
            ex.printStackTrace();
        } finally {
            if (is != null) {
                try {
                    is.close();
                } catch (Exception ignored) {
                }
            }
        }

        return glyph;
    }

    //----------------//
    // buildIconGlyph //
    //----------------//
    private Glyph buildIconGlyph (String gName)
    {
        if (logger.isFineEnabled()) {
            logger.fine("Loading icon " + gName);
        }

        Glyph       glyph = null;
        ShapeSymbol symbol = SymbolManager.getInstance()
                                          .loadSymbol(fileNameOf(gName));

        if (symbol != null) {
            glyph = new SymbolGlyph(
                symbol,
                shapeOf(gName),
                symbol.getInterline());
        }

        return glyph;
    }

    //----------------//
    // deleteXmlFiles //
    //----------------//
    private void deleteXmlFiles (File dir)
    {
        File[] files = dir.listFiles();

        for (File file : files) {
            if (FileUtil.getExtension(file)
                        .equals(FILE_EXTENSION)) {
                if (!file.delete()) {
                    logger.warning("Could not delete " + file);
                }
            }
        }
    }

    //-------------//
    // glyphNameOf //
    //-------------//
    /**
     * Build the full glyph name (which will be the unique glyph name) from the
     * file which contains the glyph description
     *
     * @param file the glyph backup file
     * @return the unique glyph name
     */
    private String glyphNameOf (File file)
    {
        return file.getParentFile()
                   .getName() + "/" + file.getName();
    }

    //-------------//
    // jaxbMarshal //
    //-------------//
    private void jaxbMarshal (Glyph        glyph,
                              OutputStream os)
        throws JAXBException, Exception
    {
        Marshaller m = getJaxbContext()
                           .createMarshaller();
        m.setProperty(Marshaller.JAXB_FORMATTED_OUTPUT, true);
        ///m.marshal(glyph, os);
        m.marshal(new GlyphValue(glyph), os);
    }

    //---------------//
    // jaxbUnmarshal //
    //---------------//
    private Glyph jaxbUnmarshal (InputStream is)
        throws JAXBException
    {
        Unmarshaller um = getJaxbContext()
                              .createUnmarshaller();

        ///return (Glyph) um.unmarshal(is);
        return new BasicGlyph((GlyphValue) um.unmarshal(is));
    }

    //----------------//
    // listLegalFiles //
    //----------------//
    private File[] listLegalFiles (File dir)
    {
        return new BlackList(dir).listFiles(glyphFilter);
    }

    //----------//
    // loadBase //
    //----------//
    /**
     * Build the map and return the collection of glyphs names in a collection
     * of directories
     *
     * @param paths the array of paths to the directories to load
     * @param monitor the observing entity if any
     * @return the collection of loaded glyphs names
     */
    private synchronized List<String> loadBase (File[]  paths,
                                                Monitor monitor)
    {
        // Files in the provided directory & its subdirectories
        List<File> files = new ArrayList<File>(4000);

        for (File path : paths) {
            loadDirectory(path, files);
        }

        if (monitor != null) {
            monitor.setTotalGlyphs(files.size());
        }

        // Now, collect the glyphs names
        List<String> base = new ArrayList<String>(4000);

        for (File file : files) {
            base.add(glyphNameOf(file));

            //            Glyph glyph = getGlyph(gName);
            //
            //            if (monitor != null) {
            //                monitor.loadedGlyph(glyph);
            //            }
        }

        ///logger.info(files.size() + " glyphs names collected");
        return base;
    }

    //--------------//
    // loadCoreBase //
    //--------------//
    /**
     * Build the collection of only the core glyphs
     *
     * @return the collection of core glyphs names
     */
    private List<String> loadCoreBase (Monitor monitor)
    {
        return loadBase(new File[] { coreFolder }, monitor);
    }

    //---------------//
    // loadDirectory //
    //---------------//
    /**
     * Retrieve recursively all files in the hierarchy starting at given
     * directory, and append them in the provided file list. If a black list
     * exists in a directory, then all black-listed files (and direct
     * sub-directories) hosted in this directory are skipped.
     *
     * @param dir the top directory where search is launched
     * @param all the list to be augmented by found files
     */
    private void loadDirectory (File       dir,
                                List<File> all)
    {
        if (logger.isFineEnabled()) {
            logger.fine("Browsing directory " + dir);
        }

        File[] files = listLegalFiles(dir);

        if (files != null) {
            for (File file : files) {
                if (file.isDirectory()) {
                    loadDirectory(file, all); // Recurse through it
                } else {
                    all.add(file);
                }
            }
        } else {
            logger.warning("Directory " + dir + " is empty");
        }
    }

    //---------------//
    // loadWholeBase //
    //---------------//
    /**
     * Build the complete map of all glyphs recorded so far, beginning by the
     * builtin icon glyphs, then the recorded glyphs.
     *
     * @return a collection of (known) glyphs names
     */
    private List<String> loadWholeBase (Monitor monitor)
    {
        return loadBase(new File[] { iconsFolder, getSheetsFolder() }, monitor);
    }

    //---------//
    // shapeOf //
    //---------//
    /**
     * Infer the shape of a glyph directly from its file name
     *
     * @param file the file that describes the glyph
     * @return the shape of the known glyph
     */
    private Shape shapeOf (File file)
    {
        try {
            // ex: THIRTY_SECOND_REST.0105.xml (for real glyphs)
            // ex: CODA.xml (for glyphs derived from icons)
            String name = FileUtil.getNameSansExtension(file);
            int    dot = name.indexOf(".");

            if (dot != -1) {
                name = name.substring(0, dot);
            }

            return Shape.valueOf(name);
        } catch (Exception ex) {
            // Not recognized
            return null;
        }
    }

    //~ Inner Interfaces -------------------------------------------------------

    //---------//
    // Monitor //
    //---------//
    /**
     * Interface <code>Monitor</code> defines the entries to a UI entity which
     * monitors the loading of glyphs by the glyph repository
     */
    public static interface Monitor
    {
        //~ Methods ------------------------------------------------------------

        /**
         * Called to pass the number of selected glyphs, which will be later
         * loaded
         *
         * @param selected the size of the selection
         */
        void setSelectedGlyphs (int selected);

        /**
         * Called to pass the total number of available glyph descriptions in
         * the training material
         *
         * @param total the size of the training material
         */
        void setTotalGlyphs (int total);

        /**
         * Called whenever a new glyph has been loaded
         *
         * @param gName the normalized glyph name
         */
        void loadedGlyph (String gName);
    }
}
