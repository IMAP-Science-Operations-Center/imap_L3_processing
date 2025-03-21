KPL/FK

Interstellar Mapping and Acceleration Probe Dynamic Frames Kernel
========================================================================

   This kernel contains SPICE frame definitions to support the IMAP mission. 
   
   This kernel is composed of primarily dynamic frames, but in general it 
   holds frame definitions for all instrument-agnostic frames, CK frames 
   used in science data processing and mapping.

Version and Date
---------------------------------------------------------------

   The TEXT_KERNEL_ID stores version information of loaded project text
   kernels.  Each entry associated with the keyword is a string that
   consists of four parts: the kernel name, version, entry date, and type.

   IMAP Dynamic Frame Kernel Version:

      \begindata

         TEXT_KERNEL_ID = 'IMAP_DYNAMIC_FRAMES V1.0.0 2024-XXXX-NN FK'

      \begintext


   Version 0.0.0 -- April 10, 2024 -- Nick Dutton (JHU/APL)

      Defined all frames from "IMAP Coordinate Frame Science.pdf"
      Used legacy FK from Parker Solar Probe as starting point


References
---------------------------------------------------------------

   1.   NAIF SPICE `Kernel Pool Required Reading'

   2.   NAIF SPICE `Frames Required Reading'

   3.   "IMAP Coordinate Frame Science.pdf"

   4.   NAIF Barycenter IDs:
      https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/naif_ids.html#Barycenters

   5.   Lagrange Points
      https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/lagrange_point/

   5.   stereo_rtn.tf, at
           ftp://sohoftp.nascom.nasa.gov/solarsoft/stereo/gen/data/spice

   6.   heliospheric.tf, at 
           ftp://sohoftp.nascom.nasa.gov/solarsoft/stereo/gen/data/spice/gen

   7.   Email from Scott Turner received May 11, 2018 containing notes 
           taken from the science team meeting on the same date.         

   8.   Snodgrass, H.B., Ulrich, R.K., 1990, Rotation of Doppler features
           in the solar photosphere. Astrophys. J. 351, 309. doi:10.1086/168467


Contact Information
---------------------------------------------------------------

   Direct questions, comments, or concerns about the contents of this
   kernel to:

      Scott Turner, JHUAPL, (443)778-1693, Scott.Turner@jhuapl.edu

   or

      Lillian Nguyen, JHUAPL (443)778-5477, Lillian.Nguyen@jhuapl.edu

   or

      Douglas Rodgers, JHUAPL (443)778-4228, Douglas.Rodgers@jhuapl.edu

   or

      Nick Dutton, JHUAPL, Nicholas.Dutton@jhuapl.edu


Implementation Notes
---------------------------------------------------------------

   This file is used by the SPICE system as follows: programs that make use
   of this frame kernel must `load' the kernel normally during program
   initialization.  Loading the kernel associates the data items with
   their names in a data structure called the `kernel pool'.  The SPICELIB
   routine FURNSH loads a kernel into the pool as shown below:

      FORTRAN: (SPICELIB)

         CALL FURNSH ( frame_kernel_name )

      C: (CSPICE)

         furnsh_c ( frame_kernel_name );

      IDL: (ICY)

         cspice_furnsh, frame_kernel_name

      MATLAB: (MICE)

         cspice_furnsh ( frame_kernel_name )

   This file was created and may be updated with a text editor or word
   processor.


IMAP Dynamic Frames
---------------------------------------------------------------

   This frame kernel defines a series of dynamic frames listed in [3]
   that support IMAP data reduction and analysis.  All 
   of the frame names defined by this kernel are prefixed with 'IMAP_' 
   to avoid conflict with alternative definitions not specific to the 
   project. Further, the project-specific ID codes 
   -43900 to -43999 have been set aside to support these dynamic frames.

   The following dynamic frames are defined in this kernel file:

      Frame Name               Relative To              Type     NAIF ID
      ======================   ===================      =======  =======

      Earth Based Frames:
      ------------------
      EARTH_FIXED              IAU_EARTH                FIXED      
      IMAP_RTN                  J2000                    DYNAMIC  -43900
      IMAP_GSE                  J2000                    DYNAMIC  -43905

      Mercury Based Frames:
      ------------------
      IMAP_MSO                  J2000                    DYNAMIC  -96903

      Venus Based Frames:
      ------------------
      IMAP_VSO                  J2000                    DYNAMIC  -96904

      Sun Based Frames:
      ------------------
      IMAP_HG                   J2000                    DYNAMIC  -96910
      IMAP_HGS                  J2000                    DYNAMIC  -43912
      IMAP_HEE                  J2000                    DYNAMIC  -43911
      IMAP_HEEQ                 J2000                    DYNAMIC  -96913
      IMAP_RTN                  J2000                    DYNAMIC  -43902
      IMAP_HERTN                J2000                    DYNAMIC  -96915      
      IMAP_HGI                  J2000                    DYNAMIC  -43909
      IMAP_HGDOPP               J2000                    DYNAMIC  -96917
      IMAP_HGMAG                J2000                    DYNAMIC  -96918
      IMAP_HGSPEC               J2000                    DYNAMIC  -96919



   \begindata

      NAIF_BODY_NAME   += ( 'IMAP_DPS' )
      NAIF_BODY_CODE   += ( -43901     )

      NAIF_BODY_NAME   += ( 'IMAP_RTN' )
      NAIF_BODY_CODE   += ( -43902     )

      NAIF_BODY_NAME   += ( 'IMAP_MDR' )
      NAIF_BODY_CODE   += ( -43903     )

      NAIF_BODY_NAME   += ( 'IMAP_MDI' )
      NAIF_BODY_CODE   += ( -43904     )

      NAIF_BODY_NAME   += ( 'IMAP_GSE' )
      NAIF_BODY_CODE   += ( -43905     )

      NAIF_BODY_NAME   += ( 'IMAP_GSM' )
      NAIF_BODY_CODE   += ( -43906     )

      NAIF_BODY_NAME   += ( 'IMAP_SM' )
      NAIF_BODY_CODE   += ( -43907     )

      NAIF_BODY_NAME   += ( 'IMAP_GEI' )
      NAIF_BODY_CODE   += ( -43908     )

      NAIF_BODY_NAME   += ( 'IMAP_HGI_J2000' )
      NAIF_BODY_CODE   += ( -43909     )

      NAIF_BODY_NAME   += ( 'IMAP_HAE' )
      NAIF_BODY_CODE   += ( -43910     )

      NAIF_BODY_NAME   += ( 'IMAP_HEE' )
      NAIF_BODY_CODE   += ( -43911     )
      
      NAIF_BODY_NAME   += ( 'IMAP_HGS' )
      NAIF_BODY_CODE   += ( -43912     )

      NAIF_BODY_NAME   += ( 'IMAP_HRE' )
      NAIF_BODY_CODE   += ( -43913     )

      NAIF_BODY_NAME   += ( 'IMAP_HNU' )
      NAIF_BODY_CODE   += ( -43914     )

      NAIF_BODY_NAME   += ( 'IMAP_GCS' )
      NAIF_BODY_CODE   += ( -43915     )



   \begintext


Earth Based Frames
---------------------------------------------------------------

   These dynamic frames are used for analyzing data in a reference
   frame tied to the dynamics of Earth.

   Some of these Earth based dynamic frames reference vectors in an
   Earth-fixed frame.  To support loading of either rotation model
   (IAU_EARTH or ITRF93), the following keywords control which model
   is used. The model is enabled by surrounding its keyword-value block 
   with the \begindata and \begintext markers (currently IAU_EARTH).

      IAU_EARTH based model:

      \begindata

         TKFRAME_EARTH_FIXED_RELATIVE = 'IAU_EARTH'
         TKFRAME_EARTH_FIXED_SPEC     = 'MATRIX'
         TKFRAME_EARTH_FIXED_MATRIX   = ( 1  0  0
                                          0  1  0
                                          0  0  1 )

      \begintext

      ITRF93 based model:

         TKFRAME_EARTH_FIXED_RELATIVE = 'ITRF93'
         TKFRAME_EARTH_FIXED_SPEC     = 'MATRIX'
         TKFRAME_EARTH_FIXED_MATRIX   = ( 1  0  0
                                          0  1  0
                                          0  0  1 )

   Note: Using the ITRF93 frame requires supplying SPICE with sufficient
         binary PCK data to cover the period of interest.  The IAU_EARTH
         frame just requires a text PCK with Earth data to be loaded.



   From [6]:

   Mean Ecliptic of Date (ECLIPDATE):

      All vectors are geometric: no aberration corrections are used.

      The X axis is the first point in Aries for the mean ecliptic of
      date, and the Z axis points along the ecliptic north pole.

      The Y axis is Z cross X, completing the right-handed reference frame.

      We freeze the frame here since many of the requested science frames
      are desired as true of date

      \begindata

         FRAME_IMAP_ECLIPDATE          = -43900
         FRAME_-43900_NAME            = 'IMAP_ECLIPDATE'
         FRAME_-43900_CLASS           = 5
         FRAME_-43900_CLASS_ID        = -43900
         FRAME_-43900_CENTER          = 399
         FRAME_-43900_RELATIVE        = 'J2000'
         FRAME_-43900_DEF_STYLE       = 'PARAMETERIZED'
         FRAME_-43900_FAMILY          = 'MEAN_ECLIPTIC_AND_EQUINOX_OF_DATE'
         FRAME_-43900_PREC_MODEL      = 'EARTH_IAU_1976'
         FRAME_-43900_OBLIQ_MODEL     = 'EARTH_IAU_1980'
         FRAME_-43900_ROTATION_STATE  = 'ROTATING'

      \begintext   

Despun Pointing Sets (DPS) Frame
============================================================================
This coordinate frame is used for ENA imager data processing and intentionally
designed for use in producing all-sky map products. 

+Z axis is parallel to the nominal spin axis of the spacecraft.  This axis
is averaged over a Pointing
Y = Z cross Necliptic where Necliptic is the unit normal to the ecliptic plane

This is a quasi-inertial reference frame

\begindata

   FRAME_IMAP_DPS              = -43901
   FRAME_-43901_NAME           = 'IMAP_DPS'
   FRAME_-43901_CLASS          = 3
   FRAME_-43901_CLASS_ID       = -43901  
   FRAME_-43901_CENTER         = -43
   CK_-43901_SCLK              = -43
   CK_-43901_SPK               = -43

\begintext






Heliographic Radial Tangential Normal (RTN) Frame
============================================================================
Standard, location-specific coordinate frame for solar and heliospheric 
  missions. 

+R (X) is the Sun to spacecraft unit vector
+T (Y) is the cross product unit vector of the Sun spin axis with the +R direction
+N (Z) is the secondary reference vector (solar rotation axis)


\begindata

   FRAME_IMAP_RTN                = -43902
   FRAME_-43902_NAME            = 'IMAP_RTN'
   FRAME_-43902_CLASS           =  5
   FRAME_-43902_CLASS_ID        =  -43902
   FRAME_-43902_CENTER          =  10
   FRAME_-43902_RELATIVE        = 'J2000'
   FRAME_-43902_DEF_STYLE       = 'PARAMETERIZED'
   FRAME_-43902_FAMILY          = 'TWO-VECTOR'
   FRAME_-43902_PRI_AXIS        = 'X'
   FRAME_-43902_PRI_VECTOR_DEF  = 'OBSERVER_TARGET_POSITION'
   FRAME_-43902_PRI_OBSERVER    = 'SUN'
   FRAME_-43902_PRI_TARGET      = 'IMAP'
   FRAME_-43902_PRI_ABCORR      = 'NONE'
   FRAME_-43902_PRI_FRAME       = 'IAU_SUN'
   FRAME_-43902_SEC_AXIS        = 'Z'
   FRAME_-43902_SEC_VECTOR_DEF  = 'CONSTANT'
   FRAME_-43902_SEC_FRAME       = 'IAU_SUN'
   FRAME_-43902_SEC_SPEC        = 'RECTANGULAR'
   FRAME_-43902_SEC_VECTOR      = ( 0, 0, 1 )


\begintext

Mission Design Rotating (MDR) Frame
==============================================================================
IMAP observatory body coordinate frame.

Origin of the frame is the Sun-Earth-Moon Barycenter L1 point defined by 
SPK in reference 5.

+X is the Sun to Earth-Moon barycenter
+Z is perpendicular to the plane of the Earth-Moon barycenter's orbit 
   around the sun; along the angular momentum vector of the Earth-Moon System
+Y completes the right handed system 

Earth-Moon Barycenter NAIF body code is 3. ref [4]:
https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/naif_ids.html#Barycenters

Lagrange Point 1 (L1) wrt Earth Barycenter is defined with naif ID 391. ref [5]:
https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/lagrange_point/

\begindata

   FRAME_IMAP_MDR                = -43903
   FRAME_-43903_NAME            = 'IMAP_MDR'
   FRAME_-43903_CLASS           =  5
   FRAME_-43903_CLASS_ID        =  -43903
   FRAME_-43903_CENTER          =  391
   FRAME_-43903_RELATIVE        = 'J2000'
   FRAME_-43903_DEF_STYLE       = 'PARAMETERIZED'
   FRAME_-43903_FAMILY          = 'TWO-VECTOR'
   FRAME_-43903_PRI_AXIS        = 'X'
   FRAME_-43903_PRI_VECTOR_DEF  = 'OBSERVER_TARGET_POSITION'
   FRAME_-43903_PRI_OBSERVER    = 'SUN'
   FRAME_-43903_PRI_TARGET      = 3
   FRAME_-43903_PRI_ABCORR      = 'NONE'
   FRAME_-43903_PRI_FRAME       = 'J2000'
   FRAME_-43903_SEC_AXIS        = 'Z'
   FRAME_-43903_SEC_VECTOR_DEF  = 'CONSTANT'
   FRAME_-43903_SEC_FRAME       = 'ECLIPJ2000'
   FRAME_-43903_SEC_SPEC        = 'RECTANGULAR'
   FRAME_-43903_SEC_VECTOR      = ( 0, 0, 1 )



\begintext


Mission Design Inertial (MDI) Frame
==============================================================================
Also called EMOJ2000 or ECLIPJ2000.  We force it to be Geocentric.

\begindata

   FRAME_IMAP_MDI              = -43904
   FRAME_-43904_NAME           = 'IMAP_MDI'
   FRAME_-43904_CLASS          = 4
   FRAME_-43904_CLASS_ID       = -43904   
   FRAME_-43904_CENTER         = 399
   TKFRAME_-43904_SPEC         = 'MATRIX'
   TKFRAME_-43904_MATRIX       = ( 1.000000,
                                   0.000000,
                                   0.000000,
                                   0.000000,
                                   1.000000,
                                   0.000000,
                                   0.000000,
                                   0.000000,
                                   1.000000 )
   TKFRAME_-43904_RELATIVE     = 'ECLIPJ2000'

\begintext


Geocentric Solar Ecliptic (GSE) Frame
==============================================================================
Rotating geocentric frame in which Sun and Earth are fixed and the XY plane
is in the ecliptic plane on specified date and time.  This is a commonly used 
frame in magnetospheric physics and space weather and is intended to be 
accurate to a specific date and time.

XY-plane is the Earth mean ecliptic (Z = Necliptic) at reference date and time

+X axis is the corresponding Earth to Sun unit vector

This is a rotating geocentric frame


\begindata

   FRAME_IMAP_GSE                = -43905
   FRAME_-43905_NAME            = 'IMAP_GSE'
   FRAME_-43905_CLASS           =  5
   FRAME_-43905_CLASS_ID        =  -43905
   FRAME_-43905_CENTER          =  399
   FRAME_-43905_RELATIVE        = 'J2000'
   FRAME_-43905_DEF_STYLE       = 'PARAMETERIZED'
   FRAME_-43905_FAMILY          = 'TWO-VECTOR'
   FRAME_-43905_PRI_AXIS        = 'X'
   FRAME_-43905_PRI_VECTOR_DEF  = 'OBSERVER_TARGET_POSITION'
   FRAME_-43905_PRI_OBSERVER    = 'EARTH'
   FRAME_-43905_PRI_TARGET      = 'SUN'
   FRAME_-43905_PRI_ABCORR      = 'NONE'
   FRAME_-43905_SEC_AXIS        = 'Z'
   FRAME_-43905_SEC_VECTOR_DEF  = 'CONSTANT'
   FRAME_-43905_SEC_FRAME       = 'IMAP_ECLIPDATE'
   FRAME_-43905_SEC_SPEC        = 'RECTANGULAR'
   FRAME_-43905_SEC_VECTOR      = ( 0, 0, 1 )

\begintext

Geocentric Solar Magnetic (GSM) Frame
==============================================================================
Rotating geocentric frame in which Sun and Earth are fixed and the XY plane
contains Earth's magnetic dipole moment.  This is a commonly used 
frame in magnetospheric physics and space weather and is intended to be 
accurate to a specific date and time.

+X axis is the corresponding Earth to Sun unit vector at reference time
+Z is the corresponding projection of the northern dipole axis onto the
GSE YZ plane

TODO: THIS IS WRONG

This is a rotating geocentric frame


\begindata

   FRAME_IMAP_GSM                = -43906
   FRAME_-43906_NAME            = 'IMAP_GSM'
   FRAME_-43906_CLASS           =  5
   FRAME_-43906_CLASS_ID        =  -43906
   FRAME_-43906_CENTER          =  399
   FRAME_-43906_RELATIVE        = 'J2000'
   FRAME_-43906_DEF_STYLE       = 'PARAMETERIZED'
   FRAME_-43906_FAMILY          = 'TWO-VECTOR'
   FRAME_-43906_PRI_AXIS        = 'X'
   FRAME_-43906_PRI_VECTOR_DEF  = 'OBSERVER_TARGET_POSITION'
   FRAME_-43906_PRI_OBSERVER    = 'EARTH'
   FRAME_-43906_PRI_TARGET      = 'SUN'
   FRAME_-43906_PRI_ABCORR      = 'NONE'
   FRAME_-43906_SEC_AXIS        = 'Z'
   FRAME_-43906_SEC_VECTOR_DEF  = 'CONSTANT'
   FRAME_-43906_SEC_FRAME       = 'IMAP_ECLIPDATE'
   FRAME_-43906_SEC_SPEC        = 'RECTANGULAR'
   FRAME_-43906_SEC_VECTOR      = ( 0, 0, 1 )



\begintext

Solar Magnetic (SM) Frame
==============================================================================
Rotating geocentric frame in which the Sun and Earth are fixed and the Zaxis
is aligned with Earth’s magnetic dipole moment. This is a commonly used 
frame in magnetospheric physics and space weather and is intended to be 
accurate to a specific date and time.

+Z axis is the Earth dipole axis in the Northern-geographic-hemisphere
   (i.e. southern magnetic dipole axis direction) at date and time
+Y axis is the cross product of +Z and Earth-Sun vector at date.

TODO: THIS IS WRONG

This is a rotating geocentric frame


\begindata

   FRAME_IMAP_SM                = -43907
   FRAME_-43907_NAME            = 'IMAP_SM'
   FRAME_-43907_CLASS           =  5
   FRAME_-43907_CLASS_ID        =  -43907
   FRAME_-43907_CENTER          =  399
   FRAME_-43907_RELATIVE        = 'J2000'
   FRAME_-43907_DEF_STYLE       = 'PARAMETERIZED'
   FRAME_-43907_FAMILY          = 'TWO-VECTOR'
   FRAME_-43907_PRI_AXIS        = 'X'
   FRAME_-43907_PRI_VECTOR_DEF  = 'OBSERVER_TARGET_POSITION'
   FRAME_-43907_PRI_OBSERVER    = 'EARTH'
   FRAME_-43907_PRI_TARGET      = 'SUN'
   FRAME_-43907_PRI_ABCORR      = 'NONE'
   FRAME_-43907_SEC_AXIS        = 'Z'
   FRAME_-43907_SEC_VECTOR_DEF  = 'CONSTANT'
   FRAME_-43907_SEC_FRAME       = 'IMAP_ECLIPDATE'
   FRAME_-43907_SEC_SPEC        = 'RECTANGULAR'
   FRAME_-43907_SEC_VECTOR      = ( 0, 0, 1 )



\begintext


Geocentric Equatorial Inertial (GEI) Frame (aka EME2000 or J2000)
==============================================================================
Just J2000 repackaged


\begindata

   FRAME_IMAP_GEI              = -43908
   FRAME_-43908_NAME           = 'IMAP_GEI'
   FRAME_-43908_CLASS          = 4
   FRAME_-43908_CLASS_ID       = -43908
   FRAME_-43908_CENTER         = 399
   TKFRAME_-43908_SPEC         = 'MATRIX'
   TKFRAME_-43908_MATRIX       = ( 1.000000,
                                   0.000000,
                                   0.000000,
                                   0.000000,
                                   1.000000,
                                   0.000000,
                                   0.000000,
                                   0.000000,
                                   1.000000 )
   TKFRAME_-43908_RELATIVE     = 'J2000'



\begintext


Heliographic Inertial (HGI) Frame at epoch J2000
==============================================================================
Rotating geocentric frame in which the Sun and Earth are fixed and the Zaxis
is aligned with Earth’s magnetic dipole moment. This is a commonly used 
frame in magnetospheric physics and space weather and is intended to be 
accurate to a specific date and time.

+Z axis is the Earth dipole axis in the Northern-geographic-hemisphere
   (i.e. southern magnetic dipole axis direction) at date and time
+Y axis is the cross product of +Z and Earth-Sun vector at date.

TODO: Not sure the Y-axis is right here...

This is an inertial frame


\begindata

   FRAME_IMAP_HGI_J2000         = -43909
   FRAME_-43909_NAME            = 'IMAP_HGI_J2000'
   FRAME_-43909_CLASS           = 5
   FRAME_-43909_CLASS_ID        = -43909
   FRAME_-43909_CENTER          = 10
   FRAME_-43909_RELATIVE        = 'J2000'
   FRAME_-43909_DEF_STYLE       = 'PARAMETERIZED'
   FRAME_-43909_FAMILY          = 'TWO-VECTOR'
   FRAME_-43909_PRI_AXIS        = 'Z'
   FRAME_-43909_PRI_VECTOR_DEF  = 'CONSTANT'
   FRAME_-43909_PRI_FRAME       = 'IAU_SUN'
   FRAME_-43909_PRI_SPEC        = 'RECTANGULAR'
   FRAME_-43909_PRI_VECTOR      = ( 0, 0, 1 )
   FRAME_-43909_SEC_AXIS        = 'Y'
   FRAME_-43909_SEC_VECTOR_DEF  = 'CONSTANT'
   FRAME_-43909_SEC_FRAME       = 'IMAP_ECLIPDATE'
   FRAME_-43909_SEC_SPEC        = 'RECTANGULAR'
   FRAME_-43909_SEC_VECTOR      = ( 0, 0, 1 )
   FRAME_-43909_FREEZE_EPOCH    = @2000-JAN-01/12:00:00



\begintext


Heliocentric Aries Ecliptic (HAE) Frame (aka ECLIPJ2000)
==============================================================================
Just ECLIPJ2000 repackaged


\begindata

   FRAME_IMAP_HAE              = -43910
   FRAME_-43910_NAME           = 'IMAP_HAE'
   FRAME_-43910_CLASS          = 4
   FRAME_-43910_CLASS_ID       = -43910
   FRAME_-43910_CENTER         = 10
   TKFRAME_-43910_SPEC         = 'MATRIX'
   TKFRAME_-43910_MATRIX       = ( 1.000000,
                                   0.000000,
                                   0.000000,
                                   0.000000,
                                   1.000000,
                                   0.000000,
                                   0.000000,
                                   0.000000,
                                   1.000000 )
   TKFRAME_-43910_RELATIVE     = 'ECLIPJ2000'



\begintext


Heliocentric Earth Ecliptic (HEE) Frame
==============================================================================


 All vectors are geometric: no aberration corrections are used.

The position of the Earth relative to the Sun is the primary vector:
the X axis points from the Sun to the Earth.

The northern surface normal to the mean ecliptic of date is the
secondary vector: the Z axis is the component of this vector
orthogonal to the X axis.

The Y axis is Z cross X, completing the right-handed reference frame.

      \begindata

         FRAME_IMAP_HEE                = -43911
         FRAME_-43911_NAME            = 'IMAP_HEE'
         FRAME_-43911_CLASS           = 5
         FRAME_-43911_CLASS_ID        = -43911
         FRAME_-43911_CENTER          = 10
         FRAME_-43911_RELATIVE        = 'J2000'
         FRAME_-43911_DEF_STYLE       = 'PARAMETERIZED'
         FRAME_-43911_FAMILY          = 'TWO-VECTOR'
         FRAME_-43911_PRI_AXIS        = 'X'
         FRAME_-43911_PRI_VECTOR_DEF  = 'OBSERVER_TARGET_POSITION'
         FRAME_-43911_PRI_OBSERVER    = 'SUN'
         FRAME_-43911_PRI_TARGET      = 'EARTH'
         FRAME_-43911_PRI_ABCORR      = 'NONE'
         FRAME_-43911_SEC_AXIS        = 'Z'
         FRAME_-43911_SEC_VECTOR_DEF  = 'CONSTANT'
         FRAME_-43911_SEC_FRAME       = 'IMAP_ECLIPDATE'
         FRAME_-43911_SEC_SPEC        = 'RECTANGULAR'
         FRAME_-43911_SEC_VECTOR      = ( 0, 0, 1 )

      \begintext

Heliographic Spherical  (HGS) Frame:
==============================================================================

All vectors are geometric: no aberration corrections are used.

+X Primary axis is defined as the ascending node on 
JAN 1, 1854 12:00:00 UTC - we achieve this with a frozen epoch realizing that
The ascending node at that time is just X in the IAU_SUN frame

The Z axis points in the solar north direction.

FREEZE EPOCH is given at JAN 1, 1854 12:00:00 UTC in TDB

The Y axis is Z cross X, completing the right-handed reference frame.

      \begindata

         FRAME_IMAP_HGS               = -43912
         FRAME_-43912_NAME            = 'IMAP_HGS'
         FRAME_-43912_CLASS           = 5
         FRAME_-43912_CLASS_ID        = -43912
         FRAME_-43912_CENTER          = 10
         FRAME_-43912_RELATIVE        = 'J2000'
         FRAME_-43912_DEF_STYLE       = 'PARAMETERIZED'
         FRAME_-43912_FAMILY          = 'TWO-VECTOR'
         FRAME_-43912_PRI_AXIS        = 'X'
         FRAME_-43912_PRI_VECTOR_DEF  = 'CONSTANT'
         FRAME_-43912_PRI_FRAME       = 'IAU_SUN'
         FRAME_-43912_PRI_SPEC        = 'RECTANGULAR'
         FRAME_-43912_PRI_VECTOR      = ( 1, 0, 0 )
         FRAME_-43912_SEC_AXIS        = 'Z'
         FRAME_-43912_SEC_VECTOR_DEF  = 'CONSTANT'
         FRAME_-43912_SEC_FRAME       = 'IAU_SUN'
         FRAME_-43912_SEC_SPEC        = 'RECTANGULAR'
         FRAME_-43912_SEC_VECTOR      = ( 0, 0, 1 )
         FRAME_-43912_FREEZE_EPOCH    = @1854-JAN-01/12:00:41.184

      \begintext


Heliospheric Ram Ecliptic (HRE) at epoch J2000
==============================================================================

All vectors are geometric: no aberration corrections are used.

+X Primary axis is defined as the nose direction

The Z axis points in the solar north direction.

The Y axis is Z cross X, completing the right-handed reference frame.

      \begindata

         FRAME_IMAP_HRE               = -43913
         FRAME_-43913_NAME            = 'IMAP_HRE'
         FRAME_-43913_CLASS           = 5
         FRAME_-43913_CLASS_ID        = -43913
         FRAME_-43913_CENTER          = 10
         FRAME_-43913_RELATIVE        = 'J2000'
         FRAME_-43913_DEF_STYLE       = 'PARAMETERIZED'
         FRAME_-43913_FAMILY          = 'TWO-VECTOR'
         FRAME_-43913_PRI_AXIS        = 'X'
         FRAME_-43913_PRI_VECTOR_DEF  = 'CONSTANT'
         FRAME_-43913_PRI_FRAME       = 'ECLIPJ2000'
         FRAME_-43913_PRI_SPEC        = 'RECTANGULAR'
         FRAME_-43913_PRI_VECTOR      = ( -0.2477, -0.9647, 0.0896 )
         FRAME_-43913_SEC_AXIS        = 'Z'
         FRAME_-43913_SEC_VECTOR_DEF  = 'CONSTANT'
         FRAME_-43913_SEC_FRAME       = 'ECLIPJ2000'
         FRAME_-43913_SEC_SPEC        = 'RECTANGULAR'
         FRAME_-43913_SEC_VECTOR      = ( 0, 0, 1 )

      \begintext


Heliospheric Nose Upfield (HNU) at epoch J2000
==============================================================================

All vectors are geometric: no aberration corrections are used.

+X Primary axis is defined as the nose direction

The Z axis is parallel to the nominal upfield direction of the ISM B-field

ref: Zirnstein et al [ApJ, 2016]

The Y axis is Z cross X, completing the right-handed reference frame.

      \begindata

         FRAME_IMAP_HNU               = -43914
         FRAME_-43914_NAME            = 'IMAP_HNU'
         FRAME_-43914_CLASS           = 5
         FRAME_-43914_CLASS_ID        = -43914
         FRAME_-43914_CENTER          = 10
         FRAME_-43914_RELATIVE        = 'J2000'
         FRAME_-43914_DEF_STYLE       = 'PARAMETERIZED'
         FRAME_-43914_FAMILY          = 'TWO-VECTOR'
         FRAME_-43914_PRI_AXIS        = 'X'
         FRAME_-43914_PRI_VECTOR_DEF  = 'CONSTANT'
         FRAME_-43914_PRI_FRAME       = 'ECLIPJ2000'
         FRAME_-43914_PRI_SPEC        = 'RECTANGULAR'
         FRAME_-43914_PRI_VECTOR      = ( -0.2477, -0.9647, 0.0896 )
         FRAME_-43914_SEC_AXIS        = 'Z'
         FRAME_-43914_SEC_VECTOR_DEF  = 'CONSTANT'
         FRAME_-43914_SEC_FRAME       = 'ECLIPJ2000'
         FRAME_-43914_SEC_SPEC        = 'RECTANGULAR'
         FRAME_-43914_SEC_VECTOR      = ( -0.5583, -0.6046, 0.5681 )

      \begintext


Galactic Coordinate System (GCS) Frame (aka GALACTIC)
==============================================================================
Just GALACTIC system II repackaged


\begindata

   FRAME_IMAP_GCS              = -43915
   FRAME_-43915_NAME           = 'IMAP_GCS'
   FRAME_-43915_CLASS          = 4
   FRAME_-43915_CLASS_ID       = -43915
   FRAME_-43915_CENTER         = 10
   TKFRAME_-43915_SPEC         = 'MATRIX'
   TKFRAME_-43915_MATRIX       = ( 1.000000,
                                   0.000000,
                                   0.000000,
                                   0.000000,
                                   1.000000,
                                   0.000000,
                                   0.000000,
                                   0.000000,
                                   1.000000 )
   TKFRAME_-43915_RELATIVE     = 'GALACTIC'



\begintext



END OF FILE
