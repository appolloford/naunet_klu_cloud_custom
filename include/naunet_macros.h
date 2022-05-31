// 
#ifndef __NAUNET_MACROS_H__
#define __NAUNET_MACROS_H__

// clang-format off
#define NAUNET_SUCCESS 0
#define NAUNET_FAIL 1

#define MAX_NSYSTEMS 1

#define NELEMENTS 7
#define NSPECIES 114
#define NHEATPROCS 0
#define NCOOLPROCS 0
#define THERMAL (NHEATPROCS || NCOOLPROCS)
#if (NSPECIES + THERMAL)
#define NEQUATIONS (NSPECIES + THERMAL)
#else
#define NEQUATIONS 1
#endif
#define NREACTIONS 1403
// non-zero terms in jacobian matrix, used in sparse matrix
#define NNZ 3028

#define IDX_ELEM_C 0
#define IDX_ELEM_H 1
#define IDX_ELEM_HE 2
#define IDX_ELEM_MG 3
#define IDX_ELEM_N 4
#define IDX_ELEM_O 5
#define IDX_ELEM_SI 6

#define IDX_GCH3OHI 0
#define IDX_GCH4I 1
#define IDX_GCOI 2
#define IDX_GCO2I 3
#define IDX_GH2CNI 4
#define IDX_GH2COI 5
#define IDX_GH2OI 6
#define IDX_GH2SiOI 7
#define IDX_GHCNI 8
#define IDX_GHNCI 9
#define IDX_GHNCOI 10
#define IDX_GHNOI 11
#define IDX_GMgI 12
#define IDX_GN2I 13
#define IDX_GNH3I 14
#define IDX_GNOI 15
#define IDX_GNO2I 16
#define IDX_GO2I 17
#define IDX_GO2HI 18
#define IDX_GSiCI 19
#define IDX_GSiC2I 20
#define IDX_GSiC3I 21
#define IDX_GSiH4I 22
#define IDX_GSiOI 23
#define IDX_CI 24
#define IDX_CII 25
#define IDX_CHI 26
#define IDX_CHII 27
#define IDX_CH2I 28
#define IDX_CH2II 29
#define IDX_CH3I 30
#define IDX_CH3II 31
#define IDX_CH3OHI 32
#define IDX_CH4I 33
#define IDX_CH4II 34
#define IDX_CNI 35
#define IDX_CNII 36
#define IDX_COI 37
#define IDX_COII 38
#define IDX_CO2I 39
#define IDX_EM 40
#define IDX_HI 41
#define IDX_HII 42
#define IDX_H2I 43
#define IDX_H2II 44
#define IDX_H2CNI 45
#define IDX_H2COI 46
#define IDX_H2COII 47
#define IDX_H2NOII 48
#define IDX_H2OI 49
#define IDX_H2OII 50
#define IDX_H2SiOI 51
#define IDX_H3II 52
#define IDX_H3COII 53
#define IDX_H3OII 54
#define IDX_HCNI 55
#define IDX_HCNII 56
#define IDX_HCNHII 57
#define IDX_HCOI 58
#define IDX_HCOII 59
#define IDX_HCO2II 60
#define IDX_HeI 61
#define IDX_HeII 62
#define IDX_HeHII 63
#define IDX_HNCI 64
#define IDX_HNCOI 65
#define IDX_HNOI 66
#define IDX_HNOII 67
#define IDX_HOCII 68
#define IDX_MgI 69
#define IDX_MgII 70
#define IDX_NI 71
#define IDX_NII 72
#define IDX_N2I 73
#define IDX_N2II 74
#define IDX_N2HII 75
#define IDX_NHI 76
#define IDX_NHII 77
#define IDX_NH2I 78
#define IDX_NH2II 79
#define IDX_NH3I 80
#define IDX_NH3II 81
#define IDX_NOI 82
#define IDX_NOII 83
#define IDX_NO2I 84
#define IDX_OI 85
#define IDX_OII 86
#define IDX_O2I 87
#define IDX_O2II 88
#define IDX_O2HI 89
#define IDX_O2HII 90
#define IDX_OCNI 91
#define IDX_OHI 92
#define IDX_OHII 93
#define IDX_SiI 94
#define IDX_SiII 95
#define IDX_SiCI 96
#define IDX_SiCII 97
#define IDX_SiC2I 98
#define IDX_SiC2II 99
#define IDX_SiC3I 100
#define IDX_SiC3II 101
#define IDX_SiHI 102
#define IDX_SiHII 103
#define IDX_SiH2I 104
#define IDX_SiH2II 105
#define IDX_SiH3I 106
#define IDX_SiH3II 107
#define IDX_SiH4I 108
#define IDX_SiH4II 109
#define IDX_SiH5II 110
#define IDX_SiOI 111
#define IDX_SiOII 112
#define IDX_SiOHII 113

#if THERMAL
#define IDX_TGAS NSPECIES
#endif

#endif