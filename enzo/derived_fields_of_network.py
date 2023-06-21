import yt
from yt import derived_field

derived_fields_map = {
    '#H2CN': 'GH2CNI_ndensity',
    '#HNC': 'GHNCI_ndensity',
    '#NO2': 'GNO2I_ndensity',
    '#SiO': 'GSiOI_ndensity',
    '#CO': 'GCOI_ndensity',
    '#HNCO': 'GHNCOI_ndensity',
    '#Mg': 'GMgI_ndensity',
    '#NO': 'GNOI_ndensity',
    '#O2': 'GO2I_ndensity',
    '#O2H': 'GO2HI_ndensity',
    '#SiC': 'GSiCI_ndensity',
    '#SiC2': 'GSiC2I_ndensity',
    '#SiC3': 'GSiC3I_ndensity',
    '#CH3OH': 'GCH3OHI_ndensity',
    '#CO2': 'GCO2I_ndensity',
    '#H2SiO': 'GH2SiOI_ndensity',
    '#HNO': 'GHNOI_ndensity',
    '#N2': 'GN2I_ndensity',
    '#H2CO': 'GH2COI_ndensity',
    '#HCN': 'GHCNI_ndensity',
    '#H2O': 'GH2OI_ndensity',
    '#NH3': 'GNH3I_ndensity',
    'SiC3+': 'SiC3II_ndensity',
    'H2CN': 'H2CNI_ndensity',
    '#CH4': 'GCH4I_ndensity',
    'H2NO+': 'H2NOII_ndensity',
    'H2SiO': 'H2SiOI_ndensity',
    'HNCO': 'HNCOI_ndensity',
    'HOC+': 'HOCII_ndensity',
    'HeH+': 'HeHII_ndensity',
    'SiC2+': 'SiC2II_ndensity',
    '#SiH4': 'GSiH4I_ndensity',
    'SiC2': 'SiC2I_ndensity',
    'SiC3': 'SiC3I_ndensity',
    'SiH5+': 'SiH5II_ndensity',
    'SiH4+': 'SiH4II_ndensity',
    'SiC+': 'SiCII_ndensity',
    'O2H': 'O2HI_ndensity',
    'SiC': 'SiCI_ndensity',
    'NO2': 'NO2I_ndensity',
    'SiH3+': 'SiH3II_ndensity',
    'SiH2+': 'SiH2II_ndensity',
    'OCN': 'OCNI_ndensity',
    'SiH2': 'SiH2I_ndensity',
    'SiOH+': 'SiOHII_ndensity',
    'SiH+': 'SiHII_ndensity',
    'SiH4': 'SiH4I_ndensity',
    'SiH': 'SiHI_ndensity',
    'SiH3': 'SiH3I_ndensity',
    'SiO+': 'SiOII_ndensity',
    'HCO2+': 'HCO2II_ndensity',
    'HNO': 'HNOI_ndensity',
    'CH3OH': 'CH3OHI_ndensity',
    'Mg': 'MgI_ndensity',
    'Mg+': 'MgII_ndensity',
    'CH4+': 'CH4II_ndensity',
    'SiO': 'SiOI_ndensity',
    'CN+': 'CNII_ndensity',
    'HCNH+': 'HCNHII_ndensity',
    'N2H+': 'N2HII_ndensity',
    'O2H+': 'O2HII_ndensity',
    'Si+': 'SiII_ndensity',
    'Si': 'SiI_ndensity',
    'HNC': 'HNCI_ndensity',
    'HNO+': 'HNOII_ndensity',
    'N2+': 'N2II_ndensity',
    'H3CO+': 'H3COII_ndensity',
    'CH4': 'CH4I_ndensity',
    'CO+': 'COII_ndensity',
    'H2+': 'H2II_ndensity',
    'NH3': 'NH3I_ndensity',
    'CH3': 'CH3I_ndensity',
    'CO2': 'CO2I_ndensity',
    'N+': 'NII_ndensity',
    'O+': 'OII_ndensity',
    'HCN+': 'HCNII_ndensity',
    'NH2+': 'NH2II_ndensity',
    'NH+': 'NHII_ndensity',
    'O2+': 'O2II_ndensity',
    'CH3+': 'CH3II_ndensity',
    'NH2': 'NH2I_ndensity',
    'CH2+': 'CH2II_ndensity',
    'H2O+': 'H2OII_ndensity',
    'NH3+': 'NH3II_ndensity',
    'NO+': 'NOII_ndensity',
    'H3O+': 'H3OII_ndensity',
    'N2': 'N2I_ndensity',
    'C+': 'CII_ndensity',
    'HCN': 'HCNI_ndensity',
    'CH+': 'CHII_ndensity',
    'CH2': 'CH2I_ndensity',
    'H2CO+': 'H2COII_ndensity',
    'NH': 'NHI_ndensity',
    'OH+': 'OHII_ndensity',
    'CN': 'CNI_ndensity',
    'H2CO': 'H2COI_ndensity',
    'HCO': 'HCOI_ndensity',
    'He+': 'HeII_ndensity',
    'CH': 'CHI_ndensity',
    'H3+': 'H3II_ndensity',
    'He': 'HeI_ndensity',
    'NO': 'NOI_ndensity',
    'N': 'NI_ndensity',
    'OH': 'OHI_ndensity',
    'O2': 'O2I_ndensity',
    'C': 'CI_ndensity',
    'H+': 'HII_ndensity',
    'HCO+': 'HCOII_ndensity',
    'H2O': 'H2OI_ndensity',
    'O': 'OI_ndensity',
    'e-': 'Electron_ndensity',
    'CO': 'COI_ndensity',
    'H2': 'H2I_ndensity',
    'H': 'HI_ndensity',
    'ElemMg': 'element_Mg_ndensity',
    'IceElemMg': 'surface_element_Mg_ndensity',
    'ElemSi': 'element_Si_ndensity',
    'IceElemSi': 'surface_element_Si_ndensity',
    'ElemHe': 'element_He_ndensity',
    'IceElemHe': 'surface_element_He_ndensity',
    'ElemN': 'element_N_ndensity',
    'IceElemN': 'surface_element_N_ndensity',
    'ElemC': 'element_C_ndensity',
    'IceElemC': 'surface_element_C_ndensity',
    'ElemO': 'element_O_ndensity',
    'IceElemO': 'surface_element_O_ndensity',
    'ElemH': 'element_H_ndensity',
    'IceElemH': 'surface_element_H_ndensity',
}

@derived_field(name='GH2CNI_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def GH2CNI_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (28.0*yt.units.mh_cgs)
    arr = num_unit*data['GH2CNI_Density']
    return arr

@derived_field(name='GHNCI_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def GHNCI_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (27.0*yt.units.mh_cgs)
    arr = num_unit*data['GHNCI_Density']
    return arr

@derived_field(name='GNO2I_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def GNO2I_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (46.0*yt.units.mh_cgs)
    arr = num_unit*data['GNO2I_Density']
    return arr

@derived_field(name='GSiOI_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def GSiOI_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (44.0*yt.units.mh_cgs)
    arr = num_unit*data['GSiOI_Density']
    return arr

@derived_field(name='GCOI_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def GCOI_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (28.0*yt.units.mh_cgs)
    arr = num_unit*data['GCOI_Density']
    return arr

@derived_field(name='GHNCOI_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def GHNCOI_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (43.0*yt.units.mh_cgs)
    arr = num_unit*data['GHNCOI_Density']
    return arr

@derived_field(name='GMgI_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def GMgI_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (24.0*yt.units.mh_cgs)
    arr = num_unit*data['GMgI_Density']
    return arr

@derived_field(name='GNOI_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def GNOI_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (30.0*yt.units.mh_cgs)
    arr = num_unit*data['GNOI_Density']
    return arr

@derived_field(name='GO2I_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def GO2I_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (32.0*yt.units.mh_cgs)
    arr = num_unit*data['GO2I_Density']
    return arr

@derived_field(name='GO2HI_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def GO2HI_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (33.0*yt.units.mh_cgs)
    arr = num_unit*data['GO2HI_Density']
    return arr

@derived_field(name='GSiCI_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def GSiCI_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (40.0*yt.units.mh_cgs)
    arr = num_unit*data['GSiCI_Density']
    return arr

@derived_field(name='GSiC2I_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def GSiC2I_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (52.0*yt.units.mh_cgs)
    arr = num_unit*data['GSiC2I_Density']
    return arr

@derived_field(name='GSiC3I_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def GSiC3I_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (64.0*yt.units.mh_cgs)
    arr = num_unit*data['GSiC3I_Density']
    return arr

@derived_field(name='GCH3OHI_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def GCH3OHI_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (32.0*yt.units.mh_cgs)
    arr = num_unit*data['GCH3OHI_Density']
    return arr

@derived_field(name='GCO2I_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def GCO2I_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (44.0*yt.units.mh_cgs)
    arr = num_unit*data['GCO2I_Density']
    return arr

@derived_field(name='GH2SiOI_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def GH2SiOI_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (46.0*yt.units.mh_cgs)
    arr = num_unit*data['GH2SiOI_Density']
    return arr

@derived_field(name='GHNOI_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def GHNOI_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (31.0*yt.units.mh_cgs)
    arr = num_unit*data['GHNOI_Density']
    return arr

@derived_field(name='GN2I_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def GN2I_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (28.0*yt.units.mh_cgs)
    arr = num_unit*data['GN2I_Density']
    return arr

@derived_field(name='GH2COI_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def GH2COI_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (30.0*yt.units.mh_cgs)
    arr = num_unit*data['GH2COI_Density']
    return arr

@derived_field(name='GHCNI_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def GHCNI_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (27.0*yt.units.mh_cgs)
    arr = num_unit*data['GHCNI_Density']
    return arr

@derived_field(name='GH2OI_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def GH2OI_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (18.0*yt.units.mh_cgs)
    arr = num_unit*data['GH2OI_Density']
    return arr

@derived_field(name='GNH3I_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def GNH3I_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (17.0*yt.units.mh_cgs)
    arr = num_unit*data['GNH3I_Density']
    return arr

@derived_field(name='SiC3II_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def SiC3II_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (64.0*yt.units.mh_cgs)
    arr = num_unit*data['SiC3II_Density']
    return arr

@derived_field(name='H2CNI_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def H2CNI_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (28.0*yt.units.mh_cgs)
    arr = num_unit*data['H2CNI_Density']
    return arr

@derived_field(name='GCH4I_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def GCH4I_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (16.0*yt.units.mh_cgs)
    arr = num_unit*data['GCH4I_Density']
    return arr

@derived_field(name='H2NOII_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def H2NOII_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (32.0*yt.units.mh_cgs)
    arr = num_unit*data['H2NOII_Density']
    return arr

@derived_field(name='H2SiOI_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def H2SiOI_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (46.0*yt.units.mh_cgs)
    arr = num_unit*data['H2SiOI_Density']
    return arr

@derived_field(name='HNCOI_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def HNCOI_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (43.0*yt.units.mh_cgs)
    arr = num_unit*data['HNCOI_Density']
    return arr

@derived_field(name='HOCII_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def HOCII_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (29.0*yt.units.mh_cgs)
    arr = num_unit*data['HOCII_Density']
    return arr

@derived_field(name='HeHII_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def HeHII_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (5.0*yt.units.mh_cgs)
    arr = num_unit*data['HeHII_Density']
    return arr

@derived_field(name='SiC2II_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def SiC2II_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (52.0*yt.units.mh_cgs)
    arr = num_unit*data['SiC2II_Density']
    return arr

@derived_field(name='GSiH4I_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def GSiH4I_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (32.0*yt.units.mh_cgs)
    arr = num_unit*data['GSiH4I_Density']
    return arr

@derived_field(name='SiC2I_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def SiC2I_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (52.0*yt.units.mh_cgs)
    arr = num_unit*data['SiC2I_Density']
    return arr

@derived_field(name='SiC3I_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def SiC3I_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (64.0*yt.units.mh_cgs)
    arr = num_unit*data['SiC3I_Density']
    return arr

@derived_field(name='SiH5II_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def SiH5II_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (33.0*yt.units.mh_cgs)
    arr = num_unit*data['SiH5II_Density']
    return arr

@derived_field(name='SiH4II_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def SiH4II_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (32.0*yt.units.mh_cgs)
    arr = num_unit*data['SiH4II_Density']
    return arr

@derived_field(name='SiCII_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def SiCII_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (40.0*yt.units.mh_cgs)
    arr = num_unit*data['SiCII_Density']
    return arr

@derived_field(name='O2HI_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def O2HI_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (33.0*yt.units.mh_cgs)
    arr = num_unit*data['O2HI_Density']
    return arr

@derived_field(name='SiCI_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def SiCI_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (40.0*yt.units.mh_cgs)
    arr = num_unit*data['SiCI_Density']
    return arr

@derived_field(name='NO2I_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def NO2I_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (46.0*yt.units.mh_cgs)
    arr = num_unit*data['NO2I_Density']
    return arr

@derived_field(name='SiH3II_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def SiH3II_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (31.0*yt.units.mh_cgs)
    arr = num_unit*data['SiH3II_Density']
    return arr

@derived_field(name='SiH2II_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def SiH2II_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (30.0*yt.units.mh_cgs)
    arr = num_unit*data['SiH2II_Density']
    return arr

@derived_field(name='OCNI_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def OCNI_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (42.0*yt.units.mh_cgs)
    arr = num_unit*data['OCNI_Density']
    return arr

@derived_field(name='SiH2I_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def SiH2I_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (30.0*yt.units.mh_cgs)
    arr = num_unit*data['SiH2I_Density']
    return arr

@derived_field(name='SiOHII_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def SiOHII_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (45.0*yt.units.mh_cgs)
    arr = num_unit*data['SiOHII_Density']
    return arr

@derived_field(name='SiHII_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def SiHII_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (29.0*yt.units.mh_cgs)
    arr = num_unit*data['SiHII_Density']
    return arr

@derived_field(name='SiH4I_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def SiH4I_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (32.0*yt.units.mh_cgs)
    arr = num_unit*data['SiH4I_Density']
    return arr

@derived_field(name='SiHI_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def SiHI_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (29.0*yt.units.mh_cgs)
    arr = num_unit*data['SiHI_Density']
    return arr

@derived_field(name='SiH3I_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def SiH3I_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (31.0*yt.units.mh_cgs)
    arr = num_unit*data['SiH3I_Density']
    return arr

@derived_field(name='SiOII_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def SiOII_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (44.0*yt.units.mh_cgs)
    arr = num_unit*data['SiOII_Density']
    return arr

@derived_field(name='HCO2II_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def HCO2II_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (45.0*yt.units.mh_cgs)
    arr = num_unit*data['HCO2II_Density']
    return arr

@derived_field(name='HNOI_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def HNOI_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (31.0*yt.units.mh_cgs)
    arr = num_unit*data['HNOI_Density']
    return arr

@derived_field(name='CH3OHI_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def CH3OHI_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (32.0*yt.units.mh_cgs)
    arr = num_unit*data['CH3OHI_Density']
    return arr

@derived_field(name='MgI_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def MgI_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (24.0*yt.units.mh_cgs)
    arr = num_unit*data['MgI_Density']
    return arr

@derived_field(name='MgII_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def MgII_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (24.0*yt.units.mh_cgs)
    arr = num_unit*data['MgII_Density']
    return arr

@derived_field(name='CH4II_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def CH4II_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (16.0*yt.units.mh_cgs)
    arr = num_unit*data['CH4II_Density']
    return arr

@derived_field(name='SiOI_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def SiOI_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (44.0*yt.units.mh_cgs)
    arr = num_unit*data['SiOI_Density']
    return arr

@derived_field(name='CNII_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def CNII_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (26.0*yt.units.mh_cgs)
    arr = num_unit*data['CNII_Density']
    return arr

@derived_field(name='HCNHII_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def HCNHII_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (28.0*yt.units.mh_cgs)
    arr = num_unit*data['HCNHII_Density']
    return arr

@derived_field(name='N2HII_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def N2HII_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (29.0*yt.units.mh_cgs)
    arr = num_unit*data['N2HII_Density']
    return arr

@derived_field(name='O2HII_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def O2HII_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (33.0*yt.units.mh_cgs)
    arr = num_unit*data['O2HII_Density']
    return arr

@derived_field(name='SiII_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def SiII_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (28.0*yt.units.mh_cgs)
    arr = num_unit*data['SiII_Density']
    return arr

@derived_field(name='SiI_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def SiI_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (28.0*yt.units.mh_cgs)
    arr = num_unit*data['SiI_Density']
    return arr

@derived_field(name='HNCI_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def HNCI_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (27.0*yt.units.mh_cgs)
    arr = num_unit*data['HNCI_Density']
    return arr

@derived_field(name='HNOII_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def HNOII_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (31.0*yt.units.mh_cgs)
    arr = num_unit*data['HNOII_Density']
    return arr

@derived_field(name='N2II_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def N2II_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (28.0*yt.units.mh_cgs)
    arr = num_unit*data['N2II_Density']
    return arr

@derived_field(name='H3COII_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def H3COII_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (31.0*yt.units.mh_cgs)
    arr = num_unit*data['H3COII_Density']
    return arr

@derived_field(name='CH4I_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def CH4I_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (16.0*yt.units.mh_cgs)
    arr = num_unit*data['CH4I_Density']
    return arr

@derived_field(name='COII_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def COII_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (28.0*yt.units.mh_cgs)
    arr = num_unit*data['COII_Density']
    return arr

@derived_field(name='H2II_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def H2II_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (2.0*yt.units.mh_cgs)
    arr = num_unit*data['H2II_Density']
    return arr

@derived_field(name='NH3I_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def NH3I_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (17.0*yt.units.mh_cgs)
    arr = num_unit*data['NH3I_Density']
    return arr

@derived_field(name='CH3I_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def CH3I_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (15.0*yt.units.mh_cgs)
    arr = num_unit*data['CH3I_Density']
    return arr

@derived_field(name='CO2I_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def CO2I_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (44.0*yt.units.mh_cgs)
    arr = num_unit*data['CO2I_Density']
    return arr

@derived_field(name='NII_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def NII_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (14.0*yt.units.mh_cgs)
    arr = num_unit*data['NII_Density']
    return arr

@derived_field(name='OII_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def OII_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (16.0*yt.units.mh_cgs)
    arr = num_unit*data['OII_Density']
    return arr

@derived_field(name='HCNII_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def HCNII_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (27.0*yt.units.mh_cgs)
    arr = num_unit*data['HCNII_Density']
    return arr

@derived_field(name='NH2II_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def NH2II_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (16.0*yt.units.mh_cgs)
    arr = num_unit*data['NH2II_Density']
    return arr

@derived_field(name='NHII_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def NHII_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (15.0*yt.units.mh_cgs)
    arr = num_unit*data['NHII_Density']
    return arr

@derived_field(name='O2II_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def O2II_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (32.0*yt.units.mh_cgs)
    arr = num_unit*data['O2II_Density']
    return arr

@derived_field(name='CH3II_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def CH3II_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (15.0*yt.units.mh_cgs)
    arr = num_unit*data['CH3II_Density']
    return arr

@derived_field(name='NH2I_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def NH2I_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (16.0*yt.units.mh_cgs)
    arr = num_unit*data['NH2I_Density']
    return arr

@derived_field(name='CH2II_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def CH2II_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (14.0*yt.units.mh_cgs)
    arr = num_unit*data['CH2II_Density']
    return arr

@derived_field(name='H2OII_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def H2OII_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (18.0*yt.units.mh_cgs)
    arr = num_unit*data['H2OII_Density']
    return arr

@derived_field(name='NH3II_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def NH3II_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (17.0*yt.units.mh_cgs)
    arr = num_unit*data['NH3II_Density']
    return arr

@derived_field(name='NOII_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def NOII_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (30.0*yt.units.mh_cgs)
    arr = num_unit*data['NOII_Density']
    return arr

@derived_field(name='H3OII_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def H3OII_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (19.0*yt.units.mh_cgs)
    arr = num_unit*data['H3OII_Density']
    return arr

@derived_field(name='N2I_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def N2I_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (28.0*yt.units.mh_cgs)
    arr = num_unit*data['N2I_Density']
    return arr

@derived_field(name='CII_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def CII_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (12.0*yt.units.mh_cgs)
    arr = num_unit*data['CII_Density']
    return arr

@derived_field(name='HCNI_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def HCNI_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (27.0*yt.units.mh_cgs)
    arr = num_unit*data['HCNI_Density']
    return arr

@derived_field(name='CHII_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def CHII_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (13.0*yt.units.mh_cgs)
    arr = num_unit*data['CHII_Density']
    return arr

@derived_field(name='CH2I_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def CH2I_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (14.0*yt.units.mh_cgs)
    arr = num_unit*data['CH2I_Density']
    return arr

@derived_field(name='H2COII_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def H2COII_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (30.0*yt.units.mh_cgs)
    arr = num_unit*data['H2COII_Density']
    return arr

@derived_field(name='NHI_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def NHI_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (15.0*yt.units.mh_cgs)
    arr = num_unit*data['NHI_Density']
    return arr

@derived_field(name='OHII_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def OHII_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (17.0*yt.units.mh_cgs)
    arr = num_unit*data['OHII_Density']
    return arr

@derived_field(name='CNI_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def CNI_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (26.0*yt.units.mh_cgs)
    arr = num_unit*data['CNI_Density']
    return arr

@derived_field(name='H2COI_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def H2COI_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (30.0*yt.units.mh_cgs)
    arr = num_unit*data['H2COI_Density']
    return arr

@derived_field(name='HCOI_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def HCOI_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (29.0*yt.units.mh_cgs)
    arr = num_unit*data['HCOI_Density']
    return arr

@derived_field(name='HeII_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def HeII_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (4.0*yt.units.mh_cgs)
    arr = num_unit*data['HeII_Density']
    return arr

@derived_field(name='CHI_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def CHI_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (13.0*yt.units.mh_cgs)
    arr = num_unit*data['CHI_Density']
    return arr

@derived_field(name='H3II_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def H3II_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (3.0*yt.units.mh_cgs)
    arr = num_unit*data['H3II_Density']
    return arr

@derived_field(name='HeI_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def HeI_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (4.0*yt.units.mh_cgs)
    arr = num_unit*data['HeI_Density']
    return arr

@derived_field(name='NOI_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def NOI_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (30.0*yt.units.mh_cgs)
    arr = num_unit*data['NOI_Density']
    return arr

@derived_field(name='NI_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def NI_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (14.0*yt.units.mh_cgs)
    arr = num_unit*data['NI_Density']
    return arr

@derived_field(name='OHI_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def OHI_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (17.0*yt.units.mh_cgs)
    arr = num_unit*data['OHI_Density']
    return arr

@derived_field(name='O2I_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def O2I_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (32.0*yt.units.mh_cgs)
    arr = num_unit*data['O2I_Density']
    return arr

@derived_field(name='CI_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def CI_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (12.0*yt.units.mh_cgs)
    arr = num_unit*data['CI_Density']
    return arr

@derived_field(name='HII_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def HII_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (1.0*yt.units.mh_cgs)
    arr = num_unit*data['HII_Density']
    return arr

@derived_field(name='HCOII_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def HCOII_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (29.0*yt.units.mh_cgs)
    arr = num_unit*data['HCOII_Density']
    return arr

@derived_field(name='H2OI_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def H2OI_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (18.0*yt.units.mh_cgs)
    arr = num_unit*data['H2OI_Density']
    return arr

@derived_field(name='OI_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def OI_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (16.0*yt.units.mh_cgs)
    arr = num_unit*data['OI_Density']
    return arr

@derived_field(name='Electron_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def Electron_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (1.0*yt.units.mh_cgs)
    arr = num_unit*data['Electron_Density']
    return arr

@derived_field(name='COI_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def COI_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (28.0*yt.units.mh_cgs)
    arr = num_unit*data['COI_Density']
    return arr

@derived_field(name='H2I_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def H2I_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (2.0*yt.units.mh_cgs)
    arr = num_unit*data['H2I_Density']
    return arr

@derived_field(name='HI_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def HI_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / (1.0*yt.units.mh_cgs)
    arr = num_unit*data['HI_Density']
    return arr

@derived_field(name='element_Mg_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def element_Mg_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    arr = (yt.YTArray(np.zeros(data.shape), '1/cm**3') +
          1*data['GMgI_ndensity'] + 1*data['MgI_ndensity'] +
          1*data['MgII_ndensity'])
    return arr

@derived_field(name='surface_element_Mg_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def surface_element_Mg_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    arr = (yt.YTArray(np.zeros(data.shape), '1/cm**3') +
          1*data['GMgI_ndensity'])
    return arr

@derived_field(name='element_Si_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def element_Si_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    arr = (yt.YTArray(np.zeros(data.shape), '1/cm**3') +
          1*data['GSiOI_ndensity'] + 1*data['GSiCI_ndensity'] +
          1*data['GSiC2I_ndensity'] + 1*data['GSiC3I_ndensity'] +
          1*data['GH2SiOI_ndensity'] + 1*data['SiC3II_ndensity'] +
          1*data['H2SiOI_ndensity'] + 1*data['SiC2II_ndensity'] +
          1*data['GSiH4I_ndensity'] + 1*data['SiC2I_ndensity'] +
          1*data['SiC3I_ndensity'] + 1*data['SiH5II_ndensity'] +
          1*data['SiH4II_ndensity'] + 1*data['SiCII_ndensity'] +
          1*data['SiCI_ndensity'] + 1*data['SiH3II_ndensity'] +
          1*data['SiH2II_ndensity'] + 1*data['SiH2I_ndensity'] +
          1*data['SiOHII_ndensity'] + 1*data['SiHII_ndensity'] +
          1*data['SiH4I_ndensity'] + 1*data['SiHI_ndensity'] +
          1*data['SiH3I_ndensity'] + 1*data['SiOII_ndensity'] +
          1*data['SiOI_ndensity'] + 1*data['SiII_ndensity'] +
          1*data['SiI_ndensity'])
    return arr

@derived_field(name='surface_element_Si_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def surface_element_Si_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    arr = (yt.YTArray(np.zeros(data.shape), '1/cm**3') +
          1*data['GSiOI_ndensity'] + 1*data['GSiCI_ndensity'] +
          1*data['GSiC2I_ndensity'] + 1*data['GSiC3I_ndensity'] +
          1*data['GH2SiOI_ndensity'] + 1*data['GSiH4I_ndensity'])
    return arr

@derived_field(name='element_He_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def element_He_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    arr = (yt.YTArray(np.zeros(data.shape), '1/cm**3') +
          1*data['HeHII_ndensity'] + 1*data['HeII_ndensity'] +
          1*data['HeI_ndensity'])
    return arr

@derived_field(name='surface_element_He_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def surface_element_He_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    arr = (yt.YTArray(np.zeros(data.shape), '1/cm**3'))
    return arr

@derived_field(name='element_N_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def element_N_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    arr = (yt.YTArray(np.zeros(data.shape), '1/cm**3') +
          1*data['GH2CNI_ndensity'] + 1*data['GHNCI_ndensity'] +
          1*data['GNO2I_ndensity'] + 1*data['GHNCOI_ndensity'] +
          1*data['GNOI_ndensity'] + 1*data['GHNOI_ndensity'] +
          2*data['GN2I_ndensity'] + 1*data['GHCNI_ndensity'] +
          1*data['GNH3I_ndensity'] + 1*data['H2CNI_ndensity'] +
          1*data['H2NOII_ndensity'] + 1*data['HNCOI_ndensity'] +
          1*data['NO2I_ndensity'] + 1*data['OCNI_ndensity'] +
          1*data['HNOI_ndensity'] + 1*data['CNII_ndensity'] +
          1*data['HCNHII_ndensity'] + 2*data['N2HII_ndensity'] +
          1*data['HNCI_ndensity'] + 1*data['HNOII_ndensity'] +
          2*data['N2II_ndensity'] + 1*data['NH3I_ndensity'] +
          1*data['NII_ndensity'] + 1*data['HCNII_ndensity'] +
          1*data['NH2II_ndensity'] + 1*data['NHII_ndensity'] +
          1*data['NH2I_ndensity'] + 1*data['NH3II_ndensity'] +
          1*data['NOII_ndensity'] + 2*data['N2I_ndensity'] +
          1*data['HCNI_ndensity'] + 1*data['NHI_ndensity'] +
          1*data['CNI_ndensity'] + 1*data['NOI_ndensity'] +
          1*data['NI_ndensity'])
    return arr

@derived_field(name='surface_element_N_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def surface_element_N_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    arr = (yt.YTArray(np.zeros(data.shape), '1/cm**3') +
          1*data['GH2CNI_ndensity'] + 1*data['GHNCI_ndensity'] +
          1*data['GNO2I_ndensity'] + 1*data['GHNCOI_ndensity'] +
          1*data['GNOI_ndensity'] + 1*data['GHNOI_ndensity'] +
          2*data['GN2I_ndensity'] + 1*data['GHCNI_ndensity'] +
          1*data['GNH3I_ndensity'])
    return arr

@derived_field(name='element_C_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def element_C_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    arr = (yt.YTArray(np.zeros(data.shape), '1/cm**3') +
          1*data['GH2CNI_ndensity'] + 1*data['GHNCI_ndensity'] +
          1*data['GCOI_ndensity'] + 1*data['GHNCOI_ndensity'] +
          1*data['GSiCI_ndensity'] + 2*data['GSiC2I_ndensity'] +
          3*data['GSiC3I_ndensity'] + 1*data['GCH3OHI_ndensity'] +
          1*data['GCO2I_ndensity'] + 1*data['GH2COI_ndensity'] +
          1*data['GHCNI_ndensity'] + 3*data['SiC3II_ndensity'] +
          1*data['H2CNI_ndensity'] + 1*data['GCH4I_ndensity'] +
          1*data['HNCOI_ndensity'] + 1*data['HOCII_ndensity'] +
          2*data['SiC2II_ndensity'] + 2*data['SiC2I_ndensity'] +
          3*data['SiC3I_ndensity'] + 1*data['SiCII_ndensity'] +
          1*data['SiCI_ndensity'] + 1*data['OCNI_ndensity'] +
          1*data['HCO2II_ndensity'] + 1*data['CH3OHI_ndensity'] +
          1*data['CH4II_ndensity'] + 1*data['CNII_ndensity'] +
          1*data['HCNHII_ndensity'] + 1*data['HNCI_ndensity'] +
          1*data['H3COII_ndensity'] + 1*data['CH4I_ndensity'] +
          1*data['COII_ndensity'] + 1*data['CH3I_ndensity'] +
          1*data['CO2I_ndensity'] + 1*data['HCNII_ndensity'] +
          1*data['CH3II_ndensity'] + 1*data['CH2II_ndensity'] +
          1*data['CII_ndensity'] + 1*data['HCNI_ndensity'] +
          1*data['CHII_ndensity'] + 1*data['CH2I_ndensity'] +
          1*data['H2COII_ndensity'] + 1*data['CNI_ndensity'] +
          1*data['H2COI_ndensity'] + 1*data['HCOI_ndensity'] +
          1*data['CHI_ndensity'] + 1*data['CI_ndensity'] +
          1*data['HCOII_ndensity'] + 1*data['COI_ndensity'])
    return arr

@derived_field(name='surface_element_C_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def surface_element_C_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    arr = (yt.YTArray(np.zeros(data.shape), '1/cm**3') +
          1*data['GH2CNI_ndensity'] + 1*data['GHNCI_ndensity'] +
          1*data['GCOI_ndensity'] + 1*data['GHNCOI_ndensity'] +
          1*data['GSiCI_ndensity'] + 2*data['GSiC2I_ndensity'] +
          3*data['GSiC3I_ndensity'] + 1*data['GCH3OHI_ndensity'] +
          1*data['GCO2I_ndensity'] + 1*data['GH2COI_ndensity'] +
          1*data['GHCNI_ndensity'] + 1*data['GCH4I_ndensity'])
    return arr

@derived_field(name='element_O_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def element_O_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    arr = (yt.YTArray(np.zeros(data.shape), '1/cm**3') +
          2*data['GNO2I_ndensity'] + 1*data['GSiOI_ndensity'] +
          1*data['GCOI_ndensity'] + 1*data['GHNCOI_ndensity'] +
          1*data['GNOI_ndensity'] + 2*data['GO2I_ndensity'] +
          2*data['GO2HI_ndensity'] + 1*data['GCH3OHI_ndensity'] +
          2*data['GCO2I_ndensity'] + 1*data['GH2SiOI_ndensity'] +
          1*data['GHNOI_ndensity'] + 1*data['GH2COI_ndensity'] +
          1*data['GH2OI_ndensity'] + 1*data['H2NOII_ndensity'] +
          1*data['H2SiOI_ndensity'] + 1*data['HNCOI_ndensity'] +
          1*data['HOCII_ndensity'] + 2*data['O2HI_ndensity'] +
          2*data['NO2I_ndensity'] + 1*data['OCNI_ndensity'] +
          1*data['SiOHII_ndensity'] + 1*data['SiOII_ndensity'] +
          2*data['HCO2II_ndensity'] + 1*data['HNOI_ndensity'] +
          1*data['CH3OHI_ndensity'] + 1*data['SiOI_ndensity'] +
          2*data['O2HII_ndensity'] + 1*data['HNOII_ndensity'] +
          1*data['H3COII_ndensity'] + 1*data['COII_ndensity'] +
          2*data['CO2I_ndensity'] + 1*data['OII_ndensity'] +
          2*data['O2II_ndensity'] + 1*data['H2OII_ndensity'] +
          1*data['NOII_ndensity'] + 1*data['H3OII_ndensity'] +
          1*data['H2COII_ndensity'] + 1*data['OHII_ndensity'] +
          1*data['H2COI_ndensity'] + 1*data['HCOI_ndensity'] +
          1*data['NOI_ndensity'] + 1*data['OHI_ndensity'] +
          2*data['O2I_ndensity'] + 1*data['HCOII_ndensity'] +
          1*data['H2OI_ndensity'] + 1*data['OI_ndensity'] +
          1*data['COI_ndensity'])
    return arr

@derived_field(name='surface_element_O_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def surface_element_O_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    arr = (yt.YTArray(np.zeros(data.shape), '1/cm**3') +
          2*data['GNO2I_ndensity'] + 1*data['GSiOI_ndensity'] +
          1*data['GCOI_ndensity'] + 1*data['GHNCOI_ndensity'] +
          1*data['GNOI_ndensity'] + 2*data['GO2I_ndensity'] +
          2*data['GO2HI_ndensity'] + 1*data['GCH3OHI_ndensity'] +
          2*data['GCO2I_ndensity'] + 1*data['GH2SiOI_ndensity'] +
          1*data['GHNOI_ndensity'] + 1*data['GH2COI_ndensity'] +
          1*data['GH2OI_ndensity'])
    return arr

@derived_field(name='element_H_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def element_H_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    arr = (yt.YTArray(np.zeros(data.shape), '1/cm**3') +
          2*data['GH2CNI_ndensity'] + 1*data['GHNCI_ndensity'] +
          1*data['GHNCOI_ndensity'] + 1*data['GO2HI_ndensity'] +
          4*data['GCH3OHI_ndensity'] + 2*data['GH2SiOI_ndensity'] +
          1*data['GHNOI_ndensity'] + 2*data['GH2COI_ndensity'] +
          1*data['GHCNI_ndensity'] + 2*data['GH2OI_ndensity'] +
          3*data['GNH3I_ndensity'] + 2*data['H2CNI_ndensity'] +
          4*data['GCH4I_ndensity'] + 2*data['H2NOII_ndensity'] +
          2*data['H2SiOI_ndensity'] + 1*data['HNCOI_ndensity'] +
          1*data['HOCII_ndensity'] + 1*data['HeHII_ndensity'] +
          4*data['GSiH4I_ndensity'] + 5*data['SiH5II_ndensity'] +
          4*data['SiH4II_ndensity'] + 1*data['O2HI_ndensity'] +
          3*data['SiH3II_ndensity'] + 2*data['SiH2II_ndensity'] +
          2*data['SiH2I_ndensity'] + 1*data['SiOHII_ndensity'] +
          1*data['SiHII_ndensity'] + 4*data['SiH4I_ndensity'] +
          1*data['SiHI_ndensity'] + 3*data['SiH3I_ndensity'] +
          1*data['HCO2II_ndensity'] + 1*data['HNOI_ndensity'] +
          4*data['CH3OHI_ndensity'] + 4*data['CH4II_ndensity'] +
          2*data['HCNHII_ndensity'] + 1*data['N2HII_ndensity'] +
          1*data['O2HII_ndensity'] + 1*data['HNCI_ndensity'] +
          1*data['HNOII_ndensity'] + 3*data['H3COII_ndensity'] +
          4*data['CH4I_ndensity'] + 2*data['H2II_ndensity'] +
          3*data['NH3I_ndensity'] + 3*data['CH3I_ndensity'] +
          1*data['HCNII_ndensity'] + 2*data['NH2II_ndensity'] +
          1*data['NHII_ndensity'] + 3*data['CH3II_ndensity'] +
          2*data['NH2I_ndensity'] + 2*data['CH2II_ndensity'] +
          2*data['H2OII_ndensity'] + 3*data['NH3II_ndensity'] +
          3*data['H3OII_ndensity'] + 1*data['HCNI_ndensity'] +
          1*data['CHII_ndensity'] + 2*data['CH2I_ndensity'] +
          2*data['H2COII_ndensity'] + 1*data['NHI_ndensity'] +
          1*data['OHII_ndensity'] + 2*data['H2COI_ndensity'] +
          1*data['HCOI_ndensity'] + 1*data['CHI_ndensity'] +
          3*data['H3II_ndensity'] + 1*data['OHI_ndensity'] +
          1*data['HII_ndensity'] + 1*data['HCOII_ndensity'] +
          2*data['H2OI_ndensity'] + 2*data['H2I_ndensity'] +
          1*data['HI_ndensity'])
    return arr

@derived_field(name='surface_element_H_ndensity', sampling_type='cell', units='1/cm**3', dimensions=yt.units.dimensions.number_density)
def surface_element_H_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    arr = (yt.YTArray(np.zeros(data.shape), '1/cm**3') +
          2*data['GH2CNI_ndensity'] + 1*data['GHNCI_ndensity'] +
          1*data['GHNCOI_ndensity'] + 1*data['GO2HI_ndensity'] +
          4*data['GCH3OHI_ndensity'] + 2*data['GH2SiOI_ndensity'] +
          1*data['GHNOI_ndensity'] + 2*data['GH2COI_ndensity'] +
          1*data['GHCNI_ndensity'] + 2*data['GH2OI_ndensity'] +
          3*data['GNH3I_ndensity'] + 4*data['GCH4I_ndensity'] +
          4*data['GSiH4I_ndensity'])
    return arr