/***********************************************************************
/
/  GRID CLASS (WRAP THE NAUNET CHEMISTRY SOLVER)
/
/  written by: Chia-Jung Hsu
/  date:       2021
/  modified1:
/
/  PURPOSE: Solve chemistry with naunet.
/
/  RETURNS:
/    SUCCESS or FAIL
/
************************************************************************/
// clang-format off
#include "preincludes.h"
#ifdef USE_NAUNET
#include "naunet_enzo.h"
#endif
#include "performance.h"
#include "macros_and_parameters.h"
#include "typedefs.h"
#include "global_data.h"
#include "Fluxes.h"
#include "GridList.h"
#include "ExternalBoundary.h"
#include "Grid.h"
#include "CosmologyParameters.h"
#include "phys_constants.h"

// function prototypes

int CosmologyComputeExpansionFactor(FLOAT time, FLOAT *a, FLOAT *dadt);
int GetUnits(float *DensityUnits, float *LengthUnits,
             float *TemperatureUnits, float *TimeUnits,
             float *VelocityUnits, FLOAT Time);
int FindField(int field, int farray[], int numfields);

int grid::NaunetWrapper()
{

#ifdef USE_NAUNET

  if (use_naunet == FALSE)
    return SUCCESS;

  if (ProcessorNumber != MyProcessorNumber)
    return SUCCESS;

  if (MultiSpecies != NAUNET_SPECIES) {
    printf("NaunetWrapper Warning: MultiSpecies = %d isn't valid for naunet. \
            Skip solving chemistry.\n", MultiSpecies);
    return SUCCESS;
  }

  LCAPERF_START("grid_NaunetWrapper");

  int GH2CNINum, GHNCINum, GNO2INum, GSiOINum, GCOINum, GHNCOINum, GMgINum,
      GNOINum, GO2INum, GO2HINum, GSiCINum, GSiC2INum, GSiC3INum, GCH3OHINum,
      GCO2INum, GH2SiOINum, GHNOINum, GN2INum, GH2COINum, GHCNINum, GH2OINum,
      GNH3INum, SiC3IINum, H2CNINum, GCH4INum, H2NOIINum, H2SiOINum, HNCOINum,
      HOCIINum, HeHIINum, SiC2IINum, GSiH4INum, SiC2INum, SiC3INum, SiH5IINum,
      SiH4IINum, SiCIINum, O2HINum, SiCINum, NO2INum, SiH3IINum, SiH2IINum,
      OCNINum, SiH2INum, SiOHIINum, SiHIINum, SiH4INum, SiHINum, SiH3INum,
      SiOIINum, HCO2IINum, HNOINum, CH3OHINum, MgINum, MgIINum, CH4IINum,
      SiOINum, CNIINum, HCNHIINum, N2HIINum, O2HIINum, SiIINum, SiINum, HNCINum,
      HNOIINum, N2IINum, H3COIINum, CH4INum, COIINum, H2IINum, NH3INum, CH3INum,
      CO2INum, NIINum, OIINum, HCNIINum, NH2IINum, NHIINum, O2IINum, CH3IINum,
      NH2INum, CH2IINum, H2OIINum, NH3IINum, NOIINum, H3OIINum, N2INum, CIINum,
      HCNINum, CHIINum, CH2INum, H2COIINum, NHINum, OHIINum, CNINum, H2COINum,
      HCOINum, HeIINum, CHINum, H3IINum, HeINum, NOINum, NINum, OHINum, O2INum,
      CINum, HIINum, HCOIINum, H2OINum, OINum, DeNum, COINum, H2INum, HINum;

  int DensNum, GENum, Vel1Num, Vel2Num, Vel3Num, TENum;

  double dt_chem = dtFixed;
  
  // Compute the size of the fields.
 
  int i, j, k, igrid;
  int size = 1;
  for (int dim = 0; dim < GridRank; dim++)
    size *= GridDimension[dim];

  int activesize = 1;
  for (int dim = 0; dim < GridRank; dim++)
    activesize *= (GridDimension[dim] - 2*NumberOfGhostZones);

  Eint32 *g_grid_dimension, *g_grid_start, *g_grid_end;
  g_grid_dimension = new Eint32[GridRank];
  g_grid_start = new Eint32[GridRank];
  g_grid_end = new Eint32[GridRank];
  for (i = 0; i < GridRank; i++) {
    g_grid_dimension[i] = (Eint32) GridDimension[i];
    g_grid_start[i] = (Eint32) GridStartIndex[i];
    g_grid_end[i] = (Eint32) GridEndIndex[i];
  }
 
  // Find fields: density, total energy, velocity1-3.
 
  if (this->IdentifyPhysicalQuantities(DensNum, GENum, Vel1Num, Vel2Num,
                                       Vel3Num, TENum) == FAIL) {
    ENZO_FAIL("Error in IdentifyPhysicalQuantities.\n");
  }

  // Find Multi-species fields.

  GH2CNINum = GHNCINum = GNO2INum = GSiOINum = GCOINum = GHNCOINum = GMgINum =
    GNOINum = GO2INum = GO2HINum = GSiCINum = GSiC2INum = GSiC3INum = GCH3OHINum
    = GCO2INum = GH2SiOINum = GHNOINum = GN2INum = GH2COINum = GHCNINum =
    GH2OINum = GNH3INum = SiC3IINum = H2CNINum = GCH4INum = H2NOIINum =
    H2SiOINum = HNCOINum = HOCIINum = HeHIINum = SiC2IINum = GSiH4INum =
    SiC2INum = SiC3INum = SiH5IINum = SiH4IINum = SiCIINum = O2HINum = SiCINum =
    NO2INum = SiH3IINum = SiH2IINum = OCNINum = SiH2INum = SiOHIINum = SiHIINum
    = SiH4INum = SiHINum = SiH3INum = SiOIINum = HCO2IINum = HNOINum = CH3OHINum
    = MgINum = MgIINum = CH4IINum = SiOINum = CNIINum = HCNHIINum = N2HIINum =
    O2HIINum = SiIINum = SiINum = HNCINum = HNOIINum = N2IINum = H3COIINum =
    CH4INum = COIINum = H2IINum = NH3INum = CH3INum = CO2INum = NIINum = OIINum
    = HCNIINum = NH2IINum = NHIINum = O2IINum = CH3IINum = NH2INum = CH2IINum =
    H2OIINum = NH3IINum = NOIINum = H3OIINum = N2INum = CIINum = HCNINum =
    CHIINum = CH2INum = H2COIINum = NHINum = OHIINum = CNINum = H2COINum =
    HCOINum = HeIINum = CHINum = H3IINum = HeINum = NOINum = NINum = OHINum =
    O2INum = CINum = HIINum = HCOIINum = H2OINum = OINum = DeNum = COINum =
    H2INum = HINum = 0;
 
  if (MultiSpecies == NAUNET_SPECIES)
    if (IdentifyNaunetSpeciesFields(GH2CNINum, GHNCINum, GNO2INum, GSiOINum,
                                    GCOINum, GHNCOINum, GMgINum, GNOINum,
                                    GO2INum, GO2HINum, GSiCINum, GSiC2INum,
                                    GSiC3INum, GCH3OHINum, GCO2INum, GH2SiOINum,
                                    GHNOINum, GN2INum, GH2COINum, GHCNINum,
                                    GH2OINum, GNH3INum, SiC3IINum, H2CNINum,
                                    GCH4INum, H2NOIINum, H2SiOINum, HNCOINum,
                                    HOCIINum, HeHIINum, SiC2IINum, GSiH4INum,
                                    SiC2INum, SiC3INum, SiH5IINum, SiH4IINum,
                                    SiCIINum, O2HINum, SiCINum, NO2INum,
                                    SiH3IINum, SiH2IINum, OCNINum, SiH2INum,
                                    SiOHIINum, SiHIINum, SiH4INum, SiHINum,
                                    SiH3INum, SiOIINum, HCO2IINum, HNOINum,
                                    CH3OHINum, MgINum, MgIINum, CH4IINum,
                                    SiOINum, CNIINum, HCNHIINum, N2HIINum,
                                    O2HIINum, SiIINum, SiINum, HNCINum,
                                    HNOIINum, N2IINum, H3COIINum, CH4INum,
                                    COIINum, H2IINum, NH3INum, CH3INum, CO2INum,
                                    NIINum, OIINum, HCNIINum, NH2IINum, NHIINum,
                                    O2IINum, CH3IINum, NH2INum, CH2IINum,
                                    H2OIINum, NH3IINum, NOIINum, H3OIINum,
                                    N2INum, CIINum, HCNINum, CHIINum, CH2INum,
                                    H2COIINum, NHINum, OHIINum, CNINum,
                                    H2COINum, HCOINum, HeIINum, CHINum, H3IINum,
                                    HeINum, NOINum, NINum, OHINum, O2INum,
                                    CINum, HIINum, HCOIINum, H2OINum, OINum,
                                    DeNum, COINum, H2INum, HINum) == FAIL) {
      ENZO_FAIL("Error in grid->IdentifyNaunetSpeciesFields.\n");
    }
 
  // Get easy to handle pointers for each variable.
 
  float *density     = BaryonField[DensNum];
  float *totalenergy = BaryonField[TENum];
  float *gasenergy   = BaryonField[GENum];
  float *velocity1   = BaryonField[Vel1Num];
  float *velocity2   = BaryonField[Vel2Num];
  float *velocity3   = BaryonField[Vel3Num];

  // Compute the cooling time.

  FLOAT a = 1.0, dadt;
  float TemperatureUnits = 1, DensityUnits = 1, LengthUnits = 1,
        VelocityUnits = 1, TimeUnits = 1, aUnits = 1;

  GetUnits(&DensityUnits, &LengthUnits, &TemperatureUnits,
           &TimeUnits, &VelocityUnits, Time);

  if (ComovingCoordinates) {
    CosmologyComputeExpansionFactor(Time+0.5*dt_chem, &a, &dadt);
    aUnits = 1.0/(1.0 + InitialRedshift);
  } 
  else if (RadiationFieldRedshift > -1){
    a        = 1.0 / (1.0 + RadiationFieldRedshift);
    aUnits   = 1.0;
  }
  float afloat = float(a);

  float *temperature = new float[size]; 
  if (this->ComputeTemperatureField(temperature) == FAIL){
    ENZO_FAIL("Error in grid->ComputeTemperatureField.");
  }

  float NumberDensityUnits = DensityUnits / mh;

  Naunet naunet;
  naunet.Init();

  // TODO: comoving, heating/cooling
  
  // Set your parameters here
  NaunetData data;

  float y[NEQUATIONS];

  for (k = GridStartIndex[2]; k <= GridEndIndex[2]; k++) {
    for (j = GridStartIndex[1]; j <= GridEndIndex[1]; j++) {
      igrid = (k * GridDimension[1] + j) * GridDimension[0] + GridStartIndex[0];
      for (i = GridStartIndex[0]; i <= GridEndIndex[0]; i++, igrid++) {

        data.nH = BaryonField[iden][i] * DensityUnits / (1.4 * mh);
        data.Tgas = temperature[igrid];

        y[IDX_GH2CNI] = max(BaryonField[GH2CNINum][igrid], 1e-40) * NumberDensityUnits / 28.0;
        y[IDX_GHNCI] = max(BaryonField[GHNCINum][igrid], 1e-40) * NumberDensityUnits / 27.0;
        y[IDX_GNO2I] = max(BaryonField[GNO2INum][igrid], 1e-40) * NumberDensityUnits / 46.0;
        y[IDX_GSiOI] = max(BaryonField[GSiOINum][igrid], 1e-40) * NumberDensityUnits / 44.0;
        y[IDX_GCOI] = max(BaryonField[GCOINum][igrid], 1e-40) * NumberDensityUnits / 28.0;
        y[IDX_GHNCOI] = max(BaryonField[GHNCOINum][igrid], 1e-40) * NumberDensityUnits / 43.0;
        y[IDX_GMgI] = max(BaryonField[GMgINum][igrid], 1e-40) * NumberDensityUnits / 24.0;
        y[IDX_GNOI] = max(BaryonField[GNOINum][igrid], 1e-40) * NumberDensityUnits / 30.0;
        y[IDX_GO2I] = max(BaryonField[GO2INum][igrid], 1e-40) * NumberDensityUnits / 32.0;
        y[IDX_GO2HI] = max(BaryonField[GO2HINum][igrid], 1e-40) * NumberDensityUnits / 33.0;
        y[IDX_GSiCI] = max(BaryonField[GSiCINum][igrid], 1e-40) * NumberDensityUnits / 40.0;
        y[IDX_GSiC2I] = max(BaryonField[GSiC2INum][igrid], 1e-40) * NumberDensityUnits / 52.0;
        y[IDX_GSiC3I] = max(BaryonField[GSiC3INum][igrid], 1e-40) * NumberDensityUnits / 64.0;
        y[IDX_GCH3OHI] = max(BaryonField[GCH3OHINum][igrid], 1e-40) * NumberDensityUnits / 32.0;
        y[IDX_GCO2I] = max(BaryonField[GCO2INum][igrid], 1e-40) * NumberDensityUnits / 44.0;
        y[IDX_GH2SiOI] = max(BaryonField[GH2SiOINum][igrid], 1e-40) * NumberDensityUnits / 46.0;
        y[IDX_GHNOI] = max(BaryonField[GHNOINum][igrid], 1e-40) * NumberDensityUnits / 31.0;
        y[IDX_GN2I] = max(BaryonField[GN2INum][igrid], 1e-40) * NumberDensityUnits / 28.0;
        y[IDX_GH2COI] = max(BaryonField[GH2COINum][igrid], 1e-40) * NumberDensityUnits / 30.0;
        y[IDX_GHCNI] = max(BaryonField[GHCNINum][igrid], 1e-40) * NumberDensityUnits / 27.0;
        y[IDX_GH2OI] = max(BaryonField[GH2OINum][igrid], 1e-40) * NumberDensityUnits / 18.0;
        y[IDX_GNH3I] = max(BaryonField[GNH3INum][igrid], 1e-40) * NumberDensityUnits / 17.0;
        y[IDX_SiC3II] = max(BaryonField[SiC3IINum][igrid], 1e-40) * NumberDensityUnits / 64.0;
        y[IDX_H2CNI] = max(BaryonField[H2CNINum][igrid], 1e-40) * NumberDensityUnits / 28.0;
        y[IDX_GCH4I] = max(BaryonField[GCH4INum][igrid], 1e-40) * NumberDensityUnits / 16.0;
        y[IDX_H2NOII] = max(BaryonField[H2NOIINum][igrid], 1e-40) * NumberDensityUnits / 32.0;
        y[IDX_H2SiOI] = max(BaryonField[H2SiOINum][igrid], 1e-40) * NumberDensityUnits / 46.0;
        y[IDX_HNCOI] = max(BaryonField[HNCOINum][igrid], 1e-40) * NumberDensityUnits / 43.0;
        y[IDX_HOCII] = max(BaryonField[HOCIINum][igrid], 1e-40) * NumberDensityUnits / 29.0;
        y[IDX_HeHII] = max(BaryonField[HeHIINum][igrid], 1e-40) * NumberDensityUnits / 5.0;
        y[IDX_SiC2II] = max(BaryonField[SiC2IINum][igrid], 1e-40) * NumberDensityUnits / 52.0;
        y[IDX_GSiH4I] = max(BaryonField[GSiH4INum][igrid], 1e-40) * NumberDensityUnits / 32.0;
        y[IDX_SiC2I] = max(BaryonField[SiC2INum][igrid], 1e-40) * NumberDensityUnits / 52.0;
        y[IDX_SiC3I] = max(BaryonField[SiC3INum][igrid], 1e-40) * NumberDensityUnits / 64.0;
        y[IDX_SiH5II] = max(BaryonField[SiH5IINum][igrid], 1e-40) * NumberDensityUnits / 33.0;
        y[IDX_SiH4II] = max(BaryonField[SiH4IINum][igrid], 1e-40) * NumberDensityUnits / 32.0;
        y[IDX_SiCII] = max(BaryonField[SiCIINum][igrid], 1e-40) * NumberDensityUnits / 40.0;
        y[IDX_O2HI] = max(BaryonField[O2HINum][igrid], 1e-40) * NumberDensityUnits / 33.0;
        y[IDX_SiCI] = max(BaryonField[SiCINum][igrid], 1e-40) * NumberDensityUnits / 40.0;
        y[IDX_NO2I] = max(BaryonField[NO2INum][igrid], 1e-40) * NumberDensityUnits / 46.0;
        y[IDX_SiH3II] = max(BaryonField[SiH3IINum][igrid], 1e-40) * NumberDensityUnits / 31.0;
        y[IDX_SiH2II] = max(BaryonField[SiH2IINum][igrid], 1e-40) * NumberDensityUnits / 30.0;
        y[IDX_OCNI] = max(BaryonField[OCNINum][igrid], 1e-40) * NumberDensityUnits / 42.0;
        y[IDX_SiH2I] = max(BaryonField[SiH2INum][igrid], 1e-40) * NumberDensityUnits / 30.0;
        y[IDX_SiOHII] = max(BaryonField[SiOHIINum][igrid], 1e-40) * NumberDensityUnits / 45.0;
        y[IDX_SiHII] = max(BaryonField[SiHIINum][igrid], 1e-40) * NumberDensityUnits / 29.0;
        y[IDX_SiH4I] = max(BaryonField[SiH4INum][igrid], 1e-40) * NumberDensityUnits / 32.0;
        y[IDX_SiHI] = max(BaryonField[SiHINum][igrid], 1e-40) * NumberDensityUnits / 29.0;
        y[IDX_SiH3I] = max(BaryonField[SiH3INum][igrid], 1e-40) * NumberDensityUnits / 31.0;
        y[IDX_SiOII] = max(BaryonField[SiOIINum][igrid], 1e-40) * NumberDensityUnits / 44.0;
        y[IDX_HCO2II] = max(BaryonField[HCO2IINum][igrid], 1e-40) * NumberDensityUnits / 45.0;
        y[IDX_HNOI] = max(BaryonField[HNOINum][igrid], 1e-40) * NumberDensityUnits / 31.0;
        y[IDX_CH3OHI] = max(BaryonField[CH3OHINum][igrid], 1e-40) * NumberDensityUnits / 32.0;
        y[IDX_MgI] = max(BaryonField[MgINum][igrid], 1e-40) * NumberDensityUnits / 24.0;
        y[IDX_MgII] = max(BaryonField[MgIINum][igrid], 1e-40) * NumberDensityUnits / 24.0;
        y[IDX_CH4II] = max(BaryonField[CH4IINum][igrid], 1e-40) * NumberDensityUnits / 16.0;
        y[IDX_SiOI] = max(BaryonField[SiOINum][igrid], 1e-40) * NumberDensityUnits / 44.0;
        y[IDX_CNII] = max(BaryonField[CNIINum][igrid], 1e-40) * NumberDensityUnits / 26.0;
        y[IDX_HCNHII] = max(BaryonField[HCNHIINum][igrid], 1e-40) * NumberDensityUnits / 28.0;
        y[IDX_N2HII] = max(BaryonField[N2HIINum][igrid], 1e-40) * NumberDensityUnits / 29.0;
        y[IDX_O2HII] = max(BaryonField[O2HIINum][igrid], 1e-40) * NumberDensityUnits / 33.0;
        y[IDX_SiII] = max(BaryonField[SiIINum][igrid], 1e-40) * NumberDensityUnits / 28.0;
        y[IDX_SiI] = max(BaryonField[SiINum][igrid], 1e-40) * NumberDensityUnits / 28.0;
        y[IDX_HNCI] = max(BaryonField[HNCINum][igrid], 1e-40) * NumberDensityUnits / 27.0;
        y[IDX_HNOII] = max(BaryonField[HNOIINum][igrid], 1e-40) * NumberDensityUnits / 31.0;
        y[IDX_N2II] = max(BaryonField[N2IINum][igrid], 1e-40) * NumberDensityUnits / 28.0;
        y[IDX_H3COII] = max(BaryonField[H3COIINum][igrid], 1e-40) * NumberDensityUnits / 31.0;
        y[IDX_CH4I] = max(BaryonField[CH4INum][igrid], 1e-40) * NumberDensityUnits / 16.0;
        y[IDX_COII] = max(BaryonField[COIINum][igrid], 1e-40) * NumberDensityUnits / 28.0;
        y[IDX_H2II] = max(BaryonField[H2IINum][igrid], 1e-40) * NumberDensityUnits / 2.0;
        y[IDX_NH3I] = max(BaryonField[NH3INum][igrid], 1e-40) * NumberDensityUnits / 17.0;
        y[IDX_CH3I] = max(BaryonField[CH3INum][igrid], 1e-40) * NumberDensityUnits / 15.0;
        y[IDX_CO2I] = max(BaryonField[CO2INum][igrid], 1e-40) * NumberDensityUnits / 44.0;
        y[IDX_NII] = max(BaryonField[NIINum][igrid], 1e-40) * NumberDensityUnits / 14.0;
        y[IDX_OII] = max(BaryonField[OIINum][igrid], 1e-40) * NumberDensityUnits / 16.0;
        y[IDX_HCNII] = max(BaryonField[HCNIINum][igrid], 1e-40) * NumberDensityUnits / 27.0;
        y[IDX_NH2II] = max(BaryonField[NH2IINum][igrid], 1e-40) * NumberDensityUnits / 16.0;
        y[IDX_NHII] = max(BaryonField[NHIINum][igrid], 1e-40) * NumberDensityUnits / 15.0;
        y[IDX_O2II] = max(BaryonField[O2IINum][igrid], 1e-40) * NumberDensityUnits / 32.0;
        y[IDX_CH3II] = max(BaryonField[CH3IINum][igrid], 1e-40) * NumberDensityUnits / 15.0;
        y[IDX_NH2I] = max(BaryonField[NH2INum][igrid], 1e-40) * NumberDensityUnits / 16.0;
        y[IDX_CH2II] = max(BaryonField[CH2IINum][igrid], 1e-40) * NumberDensityUnits / 14.0;
        y[IDX_H2OII] = max(BaryonField[H2OIINum][igrid], 1e-40) * NumberDensityUnits / 18.0;
        y[IDX_NH3II] = max(BaryonField[NH3IINum][igrid], 1e-40) * NumberDensityUnits / 17.0;
        y[IDX_NOII] = max(BaryonField[NOIINum][igrid], 1e-40) * NumberDensityUnits / 30.0;
        y[IDX_H3OII] = max(BaryonField[H3OIINum][igrid], 1e-40) * NumberDensityUnits / 19.0;
        y[IDX_N2I] = max(BaryonField[N2INum][igrid], 1e-40) * NumberDensityUnits / 28.0;
        y[IDX_CII] = max(BaryonField[CIINum][igrid], 1e-40) * NumberDensityUnits / 12.0;
        y[IDX_HCNI] = max(BaryonField[HCNINum][igrid], 1e-40) * NumberDensityUnits / 27.0;
        y[IDX_CHII] = max(BaryonField[CHIINum][igrid], 1e-40) * NumberDensityUnits / 13.0;
        y[IDX_CH2I] = max(BaryonField[CH2INum][igrid], 1e-40) * NumberDensityUnits / 14.0;
        y[IDX_H2COII] = max(BaryonField[H2COIINum][igrid], 1e-40) * NumberDensityUnits / 30.0;
        y[IDX_NHI] = max(BaryonField[NHINum][igrid], 1e-40) * NumberDensityUnits / 15.0;
        y[IDX_OHII] = max(BaryonField[OHIINum][igrid], 1e-40) * NumberDensityUnits / 17.0;
        y[IDX_CNI] = max(BaryonField[CNINum][igrid], 1e-40) * NumberDensityUnits / 26.0;
        y[IDX_H2COI] = max(BaryonField[H2COINum][igrid], 1e-40) * NumberDensityUnits / 30.0;
        y[IDX_HCOI] = max(BaryonField[HCOINum][igrid], 1e-40) * NumberDensityUnits / 29.0;
        y[IDX_HeII] = max(BaryonField[HeIINum][igrid], 1e-40) * NumberDensityUnits / 4.0;
        y[IDX_CHI] = max(BaryonField[CHINum][igrid], 1e-40) * NumberDensityUnits / 13.0;
        y[IDX_H3II] = max(BaryonField[H3IINum][igrid], 1e-40) * NumberDensityUnits / 3.0;
        y[IDX_HeI] = max(BaryonField[HeINum][igrid], 1e-40) * NumberDensityUnits / 4.0;
        y[IDX_NOI] = max(BaryonField[NOINum][igrid], 1e-40) * NumberDensityUnits / 30.0;
        y[IDX_NI] = max(BaryonField[NINum][igrid], 1e-40) * NumberDensityUnits / 14.0;
        y[IDX_OHI] = max(BaryonField[OHINum][igrid], 1e-40) * NumberDensityUnits / 17.0;
        y[IDX_O2I] = max(BaryonField[O2INum][igrid], 1e-40) * NumberDensityUnits / 32.0;
        y[IDX_CI] = max(BaryonField[CINum][igrid], 1e-40) * NumberDensityUnits / 12.0;
        y[IDX_HII] = max(BaryonField[HIINum][igrid], 1e-40) * NumberDensityUnits / 1.0;
        y[IDX_HCOII] = max(BaryonField[HCOIINum][igrid], 1e-40) * NumberDensityUnits / 29.0;
        y[IDX_H2OI] = max(BaryonField[H2OINum][igrid], 1e-40) * NumberDensityUnits / 18.0;
        y[IDX_OI] = max(BaryonField[OINum][igrid], 1e-40) * NumberDensityUnits / 16.0;
        y[IDX_eM] = max(BaryonField[DeNum][igrid], 1e-40) * NumberDensityUnits / 1.0;
        y[IDX_COI] = max(BaryonField[COINum][igrid], 1e-40) * NumberDensityUnits / 28.0;
        y[IDX_H2I] = max(BaryonField[H2INum][igrid], 1e-40) * NumberDensityUnits / 2.0;
        y[IDX_HI] = max(BaryonField[HINum][igrid], 1e-40) * NumberDensityUnits / 1.0;
        
        if (naunet.Solve(y, dt_chem * TimeUnits, &data) == NAUNET_FAIL) {    
          naunet.Finalize();
          ENZO_FAIL("Naunet failed in NaunetWrapper.C!");
        }

        BaryonField[GH2CNINum][igrid] = max(y[IDX_GH2CNI] * 28.0 / NumberDensityUnits, 1e-40);
        BaryonField[GHNCINum][igrid] = max(y[IDX_GHNCI] * 27.0 / NumberDensityUnits, 1e-40);
        BaryonField[GNO2INum][igrid] = max(y[IDX_GNO2I] * 46.0 / NumberDensityUnits, 1e-40);
        BaryonField[GSiOINum][igrid] = max(y[IDX_GSiOI] * 44.0 / NumberDensityUnits, 1e-40);
        BaryonField[GCOINum][igrid] = max(y[IDX_GCOI] * 28.0 / NumberDensityUnits, 1e-40);
        BaryonField[GHNCOINum][igrid] = max(y[IDX_GHNCOI] * 43.0 / NumberDensityUnits, 1e-40);
        BaryonField[GMgINum][igrid] = max(y[IDX_GMgI] * 24.0 / NumberDensityUnits, 1e-40);
        BaryonField[GNOINum][igrid] = max(y[IDX_GNOI] * 30.0 / NumberDensityUnits, 1e-40);
        BaryonField[GO2INum][igrid] = max(y[IDX_GO2I] * 32.0 / NumberDensityUnits, 1e-40);
        BaryonField[GO2HINum][igrid] = max(y[IDX_GO2HI] * 33.0 / NumberDensityUnits, 1e-40);
        BaryonField[GSiCINum][igrid] = max(y[IDX_GSiCI] * 40.0 / NumberDensityUnits, 1e-40);
        BaryonField[GSiC2INum][igrid] = max(y[IDX_GSiC2I] * 52.0 / NumberDensityUnits, 1e-40);
        BaryonField[GSiC3INum][igrid] = max(y[IDX_GSiC3I] * 64.0 / NumberDensityUnits, 1e-40);
        BaryonField[GCH3OHINum][igrid] = max(y[IDX_GCH3OHI] * 32.0 / NumberDensityUnits, 1e-40);
        BaryonField[GCO2INum][igrid] = max(y[IDX_GCO2I] * 44.0 / NumberDensityUnits, 1e-40);
        BaryonField[GH2SiOINum][igrid] = max(y[IDX_GH2SiOI] * 46.0 / NumberDensityUnits, 1e-40);
        BaryonField[GHNOINum][igrid] = max(y[IDX_GHNOI] * 31.0 / NumberDensityUnits, 1e-40);
        BaryonField[GN2INum][igrid] = max(y[IDX_GN2I] * 28.0 / NumberDensityUnits, 1e-40);
        BaryonField[GH2COINum][igrid] = max(y[IDX_GH2COI] * 30.0 / NumberDensityUnits, 1e-40);
        BaryonField[GHCNINum][igrid] = max(y[IDX_GHCNI] * 27.0 / NumberDensityUnits, 1e-40);
        BaryonField[GH2OINum][igrid] = max(y[IDX_GH2OI] * 18.0 / NumberDensityUnits, 1e-40);
        BaryonField[GNH3INum][igrid] = max(y[IDX_GNH3I] * 17.0 / NumberDensityUnits, 1e-40);
        BaryonField[SiC3IINum][igrid] = max(y[IDX_SiC3II] * 64.0 / NumberDensityUnits, 1e-40);
        BaryonField[H2CNINum][igrid] = max(y[IDX_H2CNI] * 28.0 / NumberDensityUnits, 1e-40);
        BaryonField[GCH4INum][igrid] = max(y[IDX_GCH4I] * 16.0 / NumberDensityUnits, 1e-40);
        BaryonField[H2NOIINum][igrid] = max(y[IDX_H2NOII] * 32.0 / NumberDensityUnits, 1e-40);
        BaryonField[H2SiOINum][igrid] = max(y[IDX_H2SiOI] * 46.0 / NumberDensityUnits, 1e-40);
        BaryonField[HNCOINum][igrid] = max(y[IDX_HNCOI] * 43.0 / NumberDensityUnits, 1e-40);
        BaryonField[HOCIINum][igrid] = max(y[IDX_HOCII] * 29.0 / NumberDensityUnits, 1e-40);
        BaryonField[HeHIINum][igrid] = max(y[IDX_HeHII] * 5.0 / NumberDensityUnits, 1e-40);
        BaryonField[SiC2IINum][igrid] = max(y[IDX_SiC2II] * 52.0 / NumberDensityUnits, 1e-40);
        BaryonField[GSiH4INum][igrid] = max(y[IDX_GSiH4I] * 32.0 / NumberDensityUnits, 1e-40);
        BaryonField[SiC2INum][igrid] = max(y[IDX_SiC2I] * 52.0 / NumberDensityUnits, 1e-40);
        BaryonField[SiC3INum][igrid] = max(y[IDX_SiC3I] * 64.0 / NumberDensityUnits, 1e-40);
        BaryonField[SiH5IINum][igrid] = max(y[IDX_SiH5II] * 33.0 / NumberDensityUnits, 1e-40);
        BaryonField[SiH4IINum][igrid] = max(y[IDX_SiH4II] * 32.0 / NumberDensityUnits, 1e-40);
        BaryonField[SiCIINum][igrid] = max(y[IDX_SiCII] * 40.0 / NumberDensityUnits, 1e-40);
        BaryonField[O2HINum][igrid] = max(y[IDX_O2HI] * 33.0 / NumberDensityUnits, 1e-40);
        BaryonField[SiCINum][igrid] = max(y[IDX_SiCI] * 40.0 / NumberDensityUnits, 1e-40);
        BaryonField[NO2INum][igrid] = max(y[IDX_NO2I] * 46.0 / NumberDensityUnits, 1e-40);
        BaryonField[SiH3IINum][igrid] = max(y[IDX_SiH3II] * 31.0 / NumberDensityUnits, 1e-40);
        BaryonField[SiH2IINum][igrid] = max(y[IDX_SiH2II] * 30.0 / NumberDensityUnits, 1e-40);
        BaryonField[OCNINum][igrid] = max(y[IDX_OCNI] * 42.0 / NumberDensityUnits, 1e-40);
        BaryonField[SiH2INum][igrid] = max(y[IDX_SiH2I] * 30.0 / NumberDensityUnits, 1e-40);
        BaryonField[SiOHIINum][igrid] = max(y[IDX_SiOHII] * 45.0 / NumberDensityUnits, 1e-40);
        BaryonField[SiHIINum][igrid] = max(y[IDX_SiHII] * 29.0 / NumberDensityUnits, 1e-40);
        BaryonField[SiH4INum][igrid] = max(y[IDX_SiH4I] * 32.0 / NumberDensityUnits, 1e-40);
        BaryonField[SiHINum][igrid] = max(y[IDX_SiHI] * 29.0 / NumberDensityUnits, 1e-40);
        BaryonField[SiH3INum][igrid] = max(y[IDX_SiH3I] * 31.0 / NumberDensityUnits, 1e-40);
        BaryonField[SiOIINum][igrid] = max(y[IDX_SiOII] * 44.0 / NumberDensityUnits, 1e-40);
        BaryonField[HCO2IINum][igrid] = max(y[IDX_HCO2II] * 45.0 / NumberDensityUnits, 1e-40);
        BaryonField[HNOINum][igrid] = max(y[IDX_HNOI] * 31.0 / NumberDensityUnits, 1e-40);
        BaryonField[CH3OHINum][igrid] = max(y[IDX_CH3OHI] * 32.0 / NumberDensityUnits, 1e-40);
        BaryonField[MgINum][igrid] = max(y[IDX_MgI] * 24.0 / NumberDensityUnits, 1e-40);
        BaryonField[MgIINum][igrid] = max(y[IDX_MgII] * 24.0 / NumberDensityUnits, 1e-40);
        BaryonField[CH4IINum][igrid] = max(y[IDX_CH4II] * 16.0 / NumberDensityUnits, 1e-40);
        BaryonField[SiOINum][igrid] = max(y[IDX_SiOI] * 44.0 / NumberDensityUnits, 1e-40);
        BaryonField[CNIINum][igrid] = max(y[IDX_CNII] * 26.0 / NumberDensityUnits, 1e-40);
        BaryonField[HCNHIINum][igrid] = max(y[IDX_HCNHII] * 28.0 / NumberDensityUnits, 1e-40);
        BaryonField[N2HIINum][igrid] = max(y[IDX_N2HII] * 29.0 / NumberDensityUnits, 1e-40);
        BaryonField[O2HIINum][igrid] = max(y[IDX_O2HII] * 33.0 / NumberDensityUnits, 1e-40);
        BaryonField[SiIINum][igrid] = max(y[IDX_SiII] * 28.0 / NumberDensityUnits, 1e-40);
        BaryonField[SiINum][igrid] = max(y[IDX_SiI] * 28.0 / NumberDensityUnits, 1e-40);
        BaryonField[HNCINum][igrid] = max(y[IDX_HNCI] * 27.0 / NumberDensityUnits, 1e-40);
        BaryonField[HNOIINum][igrid] = max(y[IDX_HNOII] * 31.0 / NumberDensityUnits, 1e-40);
        BaryonField[N2IINum][igrid] = max(y[IDX_N2II] * 28.0 / NumberDensityUnits, 1e-40);
        BaryonField[H3COIINum][igrid] = max(y[IDX_H3COII] * 31.0 / NumberDensityUnits, 1e-40);
        BaryonField[CH4INum][igrid] = max(y[IDX_CH4I] * 16.0 / NumberDensityUnits, 1e-40);
        BaryonField[COIINum][igrid] = max(y[IDX_COII] * 28.0 / NumberDensityUnits, 1e-40);
        BaryonField[H2IINum][igrid] = max(y[IDX_H2II] * 2.0 / NumberDensityUnits, 1e-40);
        BaryonField[NH3INum][igrid] = max(y[IDX_NH3I] * 17.0 / NumberDensityUnits, 1e-40);
        BaryonField[CH3INum][igrid] = max(y[IDX_CH3I] * 15.0 / NumberDensityUnits, 1e-40);
        BaryonField[CO2INum][igrid] = max(y[IDX_CO2I] * 44.0 / NumberDensityUnits, 1e-40);
        BaryonField[NIINum][igrid] = max(y[IDX_NII] * 14.0 / NumberDensityUnits, 1e-40);
        BaryonField[OIINum][igrid] = max(y[IDX_OII] * 16.0 / NumberDensityUnits, 1e-40);
        BaryonField[HCNIINum][igrid] = max(y[IDX_HCNII] * 27.0 / NumberDensityUnits, 1e-40);
        BaryonField[NH2IINum][igrid] = max(y[IDX_NH2II] * 16.0 / NumberDensityUnits, 1e-40);
        BaryonField[NHIINum][igrid] = max(y[IDX_NHII] * 15.0 / NumberDensityUnits, 1e-40);
        BaryonField[O2IINum][igrid] = max(y[IDX_O2II] * 32.0 / NumberDensityUnits, 1e-40);
        BaryonField[CH3IINum][igrid] = max(y[IDX_CH3II] * 15.0 / NumberDensityUnits, 1e-40);
        BaryonField[NH2INum][igrid] = max(y[IDX_NH2I] * 16.0 / NumberDensityUnits, 1e-40);
        BaryonField[CH2IINum][igrid] = max(y[IDX_CH2II] * 14.0 / NumberDensityUnits, 1e-40);
        BaryonField[H2OIINum][igrid] = max(y[IDX_H2OII] * 18.0 / NumberDensityUnits, 1e-40);
        BaryonField[NH3IINum][igrid] = max(y[IDX_NH3II] * 17.0 / NumberDensityUnits, 1e-40);
        BaryonField[NOIINum][igrid] = max(y[IDX_NOII] * 30.0 / NumberDensityUnits, 1e-40);
        BaryonField[H3OIINum][igrid] = max(y[IDX_H3OII] * 19.0 / NumberDensityUnits, 1e-40);
        BaryonField[N2INum][igrid] = max(y[IDX_N2I] * 28.0 / NumberDensityUnits, 1e-40);
        BaryonField[CIINum][igrid] = max(y[IDX_CII] * 12.0 / NumberDensityUnits, 1e-40);
        BaryonField[HCNINum][igrid] = max(y[IDX_HCNI] * 27.0 / NumberDensityUnits, 1e-40);
        BaryonField[CHIINum][igrid] = max(y[IDX_CHII] * 13.0 / NumberDensityUnits, 1e-40);
        BaryonField[CH2INum][igrid] = max(y[IDX_CH2I] * 14.0 / NumberDensityUnits, 1e-40);
        BaryonField[H2COIINum][igrid] = max(y[IDX_H2COII] * 30.0 / NumberDensityUnits, 1e-40);
        BaryonField[NHINum][igrid] = max(y[IDX_NHI] * 15.0 / NumberDensityUnits, 1e-40);
        BaryonField[OHIINum][igrid] = max(y[IDX_OHII] * 17.0 / NumberDensityUnits, 1e-40);
        BaryonField[CNINum][igrid] = max(y[IDX_CNI] * 26.0 / NumberDensityUnits, 1e-40);
        BaryonField[H2COINum][igrid] = max(y[IDX_H2COI] * 30.0 / NumberDensityUnits, 1e-40);
        BaryonField[HCOINum][igrid] = max(y[IDX_HCOI] * 29.0 / NumberDensityUnits, 1e-40);
        BaryonField[HeIINum][igrid] = max(y[IDX_HeII] * 4.0 / NumberDensityUnits, 1e-40);
        BaryonField[CHINum][igrid] = max(y[IDX_CHI] * 13.0 / NumberDensityUnits, 1e-40);
        BaryonField[H3IINum][igrid] = max(y[IDX_H3II] * 3.0 / NumberDensityUnits, 1e-40);
        BaryonField[HeINum][igrid] = max(y[IDX_HeI] * 4.0 / NumberDensityUnits, 1e-40);
        BaryonField[NOINum][igrid] = max(y[IDX_NOI] * 30.0 / NumberDensityUnits, 1e-40);
        BaryonField[NINum][igrid] = max(y[IDX_NI] * 14.0 / NumberDensityUnits, 1e-40);
        BaryonField[OHINum][igrid] = max(y[IDX_OHI] * 17.0 / NumberDensityUnits, 1e-40);
        BaryonField[O2INum][igrid] = max(y[IDX_O2I] * 32.0 / NumberDensityUnits, 1e-40);
        BaryonField[CINum][igrid] = max(y[IDX_CI] * 12.0 / NumberDensityUnits, 1e-40);
        BaryonField[HIINum][igrid] = max(y[IDX_HII] * 1.0 / NumberDensityUnits, 1e-40);
        BaryonField[HCOIINum][igrid] = max(y[IDX_HCOII] * 29.0 / NumberDensityUnits, 1e-40);
        BaryonField[H2OINum][igrid] = max(y[IDX_H2OI] * 18.0 / NumberDensityUnits, 1e-40);
        BaryonField[OINum][igrid] = max(y[IDX_OI] * 16.0 / NumberDensityUnits, 1e-40);
        BaryonField[DeNum][igrid] = max(y[IDX_eM] * 1.0 / NumberDensityUnits, 1e-40);
        BaryonField[COINum][igrid] = max(y[IDX_COI] * 28.0 / NumberDensityUnits, 1e-40);
        BaryonField[H2INum][igrid] = max(y[IDX_H2I] * 2.0 / NumberDensityUnits, 1e-40);
        BaryonField[HINum][igrid] = max(y[IDX_HI] * 1.0 / NumberDensityUnits, 1e-40);
        
      }
    }
  }
  
  
  naunet.Finalize();

  delete [] temperature;

  delete [] g_grid_dimension;
  delete [] g_grid_start;
  delete [] g_grid_end;

  LCAPERF_STOP("grid_NaunetWrapper");
#endif

  return SUCCESS;
}