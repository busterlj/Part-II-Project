//----------------------------------------------------------------------------//
//                                                                            //
//                      G l y p h C o m p o s i t i o n                       //
//                                                                            //
//----------------------------------------------------------------------------//
// <editor-fold defaultstate="collapsed" desc="hdr">                          //
//  Copyright (C) Herve Bitteur 2000-2010. All rights reserved.               //
//  This software is released under the GNU General Public License.           //
//  Goto http://kenai.com/projects/audiveris to report bugs or suggestions.   //
//----------------------------------------------------------------------------//
// </editor-fold>
package omr.glyph.facets;

import omr.check.Result;

import omr.glyph.GlyphSection;

import omr.sheet.SystemInfo;

import java.util.Collection;
import java.util.Set;
import java.util.SortedSet;

/**
 * Interface {@code GlyphComposition} defines the facet that handles the way a
 * glyph is composed of sections members, as well as the relationships with
 * sub-glyphs (parts) if any.
 *
 *
 * @author Hervé Bitteur
 */
interface GlyphComposition
    extends GlyphFacet
{
    //~ Enumerations -----------------------------------------------------------

    /** Specifies whether a section must point back to a containing glyph */
    enum Linking {
        //~ Enumeration constant initializers ----------------------------------


        /** Make the section point back to the containing glyph */
        LINK_BACK,
        /** Do not make the section point back to the containing glyph */
        NO_LINK_BACK;
    }

    //~ Methods ----------------------------------------------------------------

    //-----------//
    // setPartOf //
    //-----------//
    /**
     * Record the link from this glyph as part of a larger compound
     *
     * @param compound the containing compound
     */
    public void setPartOf (Glyph compound);

    //-----------//
    // getPartOf //
    //-----------//
    /**
     * Report the containing compound, if any
     *
     * @return compound the containing compound if any
     */
    public Glyph getPartOf ();

    //----------//
    // setParts //
    //----------//
    /**
     * Record the parts that compose this compound gmyph
     *
     * @param parts the contained parts
     */
    public void setParts (Collection<?extends Glyph> parts);

    //----------//
    // getParts //
    //----------//
    /**
     * Report the parts, if any, that compose this compound
     * @return the set of glyphs, perhaps empty, but never null
     */
    public Set<Glyph> getParts ();

    //----------//
    // isActive //
    //----------//
    /**
     * Tests whether this glyph is active (all its member sections point to it)
     * @return true if glyph is active, false otherwise
     */
    boolean isActive ();

    //----------------//
    // getAlienSystem //
    //----------------//
    /**
     * Check whether all the glyph sections belong to the same system
     * @param system the supposed containing system
     * @return the alien system found, or null if OK
     */
    SystemInfo getAlienSystem (SystemInfo system);

    //-----------------//
    // getFirstSection //
    //-----------------//
    /**
     * Report the first section in the ordered collection of glyph members
     *
     * @return the first section of the glyph
     */
    GlyphSection getFirstSection ();

    //------------//
    // getMembers //
    //------------//
    /**
     * Report the set of member sections.
     *
     * @return member sections
     */
    SortedSet<GlyphSection> getMembers ();

    //-----------//
    // setResult //
    //-----------//
    /**
     * Record the analysis result in the glyph itself
     *
     * @param result the assigned result
     */
    void setResult (Result result);

    //-----------//
    // getResult //
    //-----------//
    /**
     * Report the result found during analysis of this glyph
     *
     * @return the analysis result
     */
    Result getResult ();

    //--------------//
    // isSuccessful //
    //--------------//
    /**
     * Convenient method to check whether the glyph is successfully recognized
     * @return true if the glyph is successfully recognized
     */
    boolean isSuccessful ();

    //------------------//
    // addGlyphSections //
    //------------------//
    /**
     * Add another glyph (with its sections of points) to this one
     *
     * @param other The merged glyph
     * @param linkSections Should we set the link from sections to glyph ?
     */
    void addGlyphSections (Glyph   other,
                           Linking linkSections);

    //------------//
    // addSection //
    //------------//
    /**
     * Add a section as a member of this glyph.
     *
     * @param section The section to be included
     * @param link While adding a section to this glyph members, should we also
     *             set the link from section back to the glyph?
     */
    void addSection (GlyphSection section,
                     Linking      link);

    //-------------//
    // cutSections //
    //-------------//
    /**
     * Cut the link to this glyph from its member sections, only if the sections
     * actually point to this glyph
     */
    void cutSections ();

    //-----------------//
    // linkAllSections //
    //-----------------//
    /**
     * Make all the glyph's sections point back to this glyph
     */
    void linkAllSections ();
}
