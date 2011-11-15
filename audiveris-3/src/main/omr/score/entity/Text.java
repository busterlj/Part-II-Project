//----------------------------------------------------------------------------//
//                                                                            //
//                                  T e x t                                   //
//                                                                            //
//----------------------------------------------------------------------------//
// <editor-fold defaultstate="collapsed" desc="hdr">                          //
//  Copyright (C) Herve Bitteur 2000-2010. All rights reserved.               //
//  This software is released under the GNU General Public License.           //
//  Goto http://kenai.com/projects/audiveris to report bugs or suggestions.   //
//----------------------------------------------------------------------------//
// </editor-fold>
package omr.score.entity;

import omr.glyph.facets.Glyph;
import omr.glyph.text.Sentence;
import omr.glyph.text.TextInfo;
import omr.glyph.text.TextRole;

import omr.log.Logger;

import omr.score.common.SystemPoint;
import omr.score.common.SystemRectangle;
import omr.score.visitor.ScoreVisitor;

import java.awt.Font;

/**
 * Class <code>Text</code> handles any textual score entity.
 *
 * <p><b>Nota</b>: There is exactly one Text entity per sentence, except for
 * lyrics items for which we build one LyricsItem (subclass of Text) for each
 * textual glyph. The reason is that, except for lyrics, only the full sentence
 * is meaningful: for example "Ludwig van Beethoven" is meaningful as a Creator
 * Text, but the various glyphs "Ludwig", "van", "Beethoven" are not.
 * For lyrics, since we can have very long sentences, and since the positioning
 * of every syllable must be done with precision, we handle one LyricsItem Text
 * entity per isolated textual glyph.</p>
 *
 * <p>Working at the sentence level also allows a better accuracy in the setting
 * of parameters (such as baseline or font) for the whole sentence.</p>
 *
 * <h4>Synoptic of Text Translation:<br/>
 *    <img src="doc-files/TextTranslation.jpg"/>
 * </h4>
 *
 * @author Hervé Bitteur
 */
public abstract class Text
    extends PartNode
{
    //~ Static fields/initializers ---------------------------------------------

    /** Usual logger utility */
    private static final Logger logger = Logger.getLogger(Text.class);

    //~ Instance fields --------------------------------------------------------

    /** The containing sentence */
    private final Sentence sentence;

    /** The font to render this text entity */
    protected Font font;

    //~ Constructors -----------------------------------------------------------

    //------//
    // Text //
    //------//
    /**
     * Creates a new Text object.
     *
     * @param sentence the larger sentence
     */
    public Text (Sentence sentence)
    {
        this(sentence, sentence.getLocation());
    }

    //------//
    // Text //
    //------//
    /**
     * Creates a new Text object, with a specific location, different from the
     * sentence location
     *
     * @param sentence the larger sentence
     * @param location specific location
     */
    public Text (Sentence    sentence,
                 SystemPoint location)
    {
        super(sentence.getSystemPart());
        this.sentence = sentence;
        setReferencePoint(location);

        setBox(sentence.getSystemContour());

        // Proper font ?
        if (sentence.getFontSize() != null) {
            font = TextInfo.basicFont.deriveFont(
                (float) sentence.getFontSize());
        } else {
            font = TextInfo.basicFont;
        }

        ///determineFontSize();
        if (logger.isFineEnabled()) {
            logger.fine("Created " + this);
        }
    }

    //~ Methods ----------------------------------------------------------------

    //------------//
    // getContent //
    //------------//
    /**
     * Report the current string value of this text
     * @return the string value of this text
     */
    public String getContent ()
    {
        String str = sentence.getTextContent();

        if (str == null) {
            str = sentence.getContentFromItems();
        }

        return str;
    }

    //-------------------//
    // getLyricsFontSize //
    //-------------------//
    /**
     * Report the font size to be exported for the lyrics
     *
     * @return the exported lyrics font size
     */
    public static int getLyricsFontSize ()
    {
        return TextInfo.basicFont.getSize();
    }

    //-------------//
    // getFontSize //
    //-------------//
    /**
     * Report the font size to be exported for this text
     *
     * @return the exported font size
     */
    public int getFontSize ()
    {
        return font.getSize();
    }

    //---------------//
    // getLyricsFont //
    //---------------//
    /**
     * Report the font to be used for handling lyrics text
     * @return the lyrics text font
     */
    public static Font getLyricsFont ()
    {
        return TextInfo.basicFont;
    }

    //-------------//
    // getSentence //
    //-------------//
    /**
     * Report the sentence that relates to this text
     * @return the related sentence
     */
    public Sentence getSentence ()
    {
        return sentence;
    }

    //----------//
    // getWidth //
    //----------//
    /**
     * Report the width in units of this text
     *
     * @return the text width in units
     */
    public int getWidth ()
    {
        return getBox().width;
    }

    //--------//
    // accept //
    //--------//
    @Override
    public boolean accept (ScoreVisitor visitor)
    {
        return visitor.visit(this);
    }

    //----------//
    // populate //
    //----------//
    /**
     * Allocate the proper score entity (or entities) that correspond to this
     * textual sentence. This word or sequence of words may be: <ul>
     * <li>a Direction</li>
     * <li>a part Name</li>
     * <li>a part Number</li>
     * <li>a Creator</li>
     * <li>a Copyright</li>
     * <li>one or several LyricsItem entities</li>
     *
     * @param sentence the whole sentence
     * @param location its starting reference wrt containing system
     */
    public static void populate (Sentence    sentence,
                                 SystemPoint location)
    {
        final SystemPart  systemPart = sentence.getSystemPart();
        final ScoreSystem system = systemPart.getSystem();
        final TextRole    role = sentence.getTextRole();

        if ((role == null) || (role == TextRole.Unknown)) {
            systemPart.addError(
                sentence.getGlyphs().first(),
                "Sentence with no role defined");
        }

        if (logger.isFineEnabled()) {
            logger.fine(
                "Populating " + sentence + " " + role + " \"" +
                sentence.getTextContent() + "\"");
        }

        if (role == null) {
            return;
        }

        switch (role) {
        case Lyrics :

            // Create as many lyrics items as needed
            for (Glyph item : sentence.getGlyphs()) {
                SystemRectangle itemBox = system.toSystemRectangle(
                    item.getContourBox());
                String          itemStr = item.getTextInfo()
                                              .getContent();

                if (itemStr == null) {
                    int nbChar = (int) Math.rint(
                        (double) itemBox.width / sentence.getTextHeight());
                    itemStr = role.getStringHolder(nbChar);
                }

                item.setTranslation(
                    new LyricsItem(
                        sentence,
                        new SystemPoint(itemBox.x, location.y),
                        item,
                        itemBox.width,
                        itemStr));
            }

            break;

        case Title :
            sentence.setGlyphsTranslation(new TitleText(sentence));

            break;

        case Direction :

            Measure measure = systemPart.getMeasureAt(location);
            sentence.setGlyphsTranslation(
                new DirectionStatement(
                    measure,
                    location,
                    measure.getDirectionChord(location),
                    sentence,
                    new DirectionText(sentence)));

            break;

        case Number :
            sentence.setGlyphsTranslation(new NumberText(sentence));

            break;

        case Name :
            sentence.setGlyphsTranslation(new NameText(sentence));

            break;

        case Creator :
            sentence.setGlyphsTranslation(new CreatorText(sentence));

            break;

        case Rights :
            sentence.setGlyphsTranslation(new RightsText(sentence));

            break;

        default :
            sentence.setGlyphsTranslation(new DefaultText(sentence));
        }
    }

    //---------//
    // getFont //
    //---------//
    /**
     * Report the font to render this text
     *
     * @return the font to render this text
     */
    public Font getFont ()
    {
        return font;
    }

    //----------//
    // toString //
    //----------//
    /**
     * {@inheritDoc}
     */
    @Override
    public String toString ()
    {
        StringBuilder sb = new StringBuilder();
        sb.append("{Text ")
          .append(sentence.getTextRole())
          .append(internalsString());

        sb.append(" font:")
          .append(font.getSize());

        if (getContent() != null) {
            sb.append(" \"")
              .append(getContent())
              .append("\"");
        }

        sb.append(" loc:")
          .append(getReferencePoint());

        sb.append(" S")
          .append(getSystem().getId())
          .append("P")
          .append(getPart().getId());
        sb.append("}");

        return sb.toString();
    }

    //-----------------------//
    // computeReferencePoint //
    //-----------------------//
    @Override
    protected void computeReferencePoint ()
    {
        setReferencePoint(sentence.getLocation());
    }

    //-----------------//
    // internalsString //
    //-----------------//
    /**
     * Return the string of the internals of this class, for inclusion in a
     * toString
     *
     * @return the string of internals
     */
    protected String internalsString ()
    {
        return ""; // By default
    }

    //~ Inner Classes ----------------------------------------------------------

    //-------------//
    // CreatorText //
    //-------------//
    /** Subclass of Text, dedicated to a Creator (composer, lyricist, ...) */
    public static class CreatorText
        extends Text
    {
        //~ Enumerations -------------------------------------------------------

        public enum CreatorType {
            //~ Enumeration constant initializers ------------------------------

            arranger,composer, lyricist,
            poet,
            transcriber,
            translator;
        }

        //~ Instance fields ----------------------------------------------------

        /** Creator type, if any */
        private CreatorType creatorType;

        //~ Constructors -------------------------------------------------------

        public CreatorText (Sentence sentence)
        {
            super(sentence);
            setCreatorType(sentence.getTextType());
        }

        //~ Methods ------------------------------------------------------------

        public void setCreatorType (CreatorType creatorType)
        {
            this.creatorType = creatorType;
        }

        public CreatorType getCreatorType ()
        {
            return creatorType;
        }

        @Override
        protected String internalsString ()
        {
            if (creatorType != null) {
                return " " + creatorType;
            } else {
                return "";
            }
        }
    }

    //-------------//
    // DefaultText //
    //-------------//
    /**
     * Subclass of Text, with no precise role assigned.
     * Perhaps, we could get rid of this class
     */
    public static class DefaultText
        extends Text
    {
        //~ Constructors -------------------------------------------------------

        public DefaultText (Sentence sentence)
        {
            super(sentence);
        }
    }

    //---------------//
    // DirectionText //
    //---------------//
    /** Subclass of Text, dedicated to a Direction instruction */
    public static class DirectionText
        extends Text
    {
        //~ Constructors -------------------------------------------------------

        public DirectionText (Sentence sentence)
        {
            super(sentence);
        }
    }

    //----------//
    // NameText //
    //----------//
    /** Subclass of Text, dedicated to a part Name */
    public static class NameText
        extends Text
    {
        //~ Constructors -------------------------------------------------------

        public NameText (Sentence sentence)
        {
            super(sentence);

            if (getContent() != null) {
                sentence.getSystemPart()
                        .getScorePart()
                        .setName(getContent());
            }
        }
    }

    //------------//
    // NumberText //
    //------------//
    /** Subclass of Text, dedicated to a score Number */
    public static class NumberText
        extends Text
    {
        //~ Constructors -------------------------------------------------------

        public NumberText (Sentence sentence)
        {
            super(sentence);
        }
    }

    //------------//
    // RightsText //
    //------------//
    /** Subclass of Text, dedicated to a copyright statement */
    public static class RightsText
        extends Text
    {
        //~ Constructors -------------------------------------------------------

        public RightsText (Sentence sentence)
        {
            super(sentence);
        }
    }

    //-----------//
    // TitleText //
    //-----------//
    /** Subclass of Text, dedicated to a score Title */
    public static class TitleText
        extends Text
    {
        //~ Constructors -------------------------------------------------------

        public TitleText (Sentence sentence)
        {
            super(sentence);
        }
    }
}
