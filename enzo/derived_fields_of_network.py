import yt
from yt import derived_field

mh_cgs = float(yt.units.mh_cgs)

@derived_field(name='GCH3OHI_ndensity', sampling_type='cell')
def GCH3OHI_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 32.0
    arr = (num_unit*data['GCH3OHI_Density']).to_ndarray()
    return arr

@derived_field(name='GCH4I_ndensity', sampling_type='cell')
def GCH4I_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 16.0
    arr = (num_unit*data['GCH4I_Density']).to_ndarray()
    return arr

@derived_field(name='GCOI_ndensity', sampling_type='cell')
def GCOI_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 28.0
    arr = (num_unit*data['GCOI_Density']).to_ndarray()
    return arr

@derived_field(name='GCO2I_ndensity', sampling_type='cell')
def GCO2I_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 44.0
    arr = (num_unit*data['GCO2I_Density']).to_ndarray()
    return arr

@derived_field(name='GH2CNI_ndensity', sampling_type='cell')
def GH2CNI_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 28.0
    arr = (num_unit*data['GH2CNI_Density']).to_ndarray()
    return arr

@derived_field(name='GH2COI_ndensity', sampling_type='cell')
def GH2COI_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 30.0
    arr = (num_unit*data['GH2COI_Density']).to_ndarray()
    return arr

@derived_field(name='GH2OI_ndensity', sampling_type='cell')
def GH2OI_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 18.0
    arr = (num_unit*data['GH2OI_Density']).to_ndarray()
    return arr

@derived_field(name='GH2SiOI_ndensity', sampling_type='cell')
def GH2SiOI_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 46.0
    arr = (num_unit*data['GH2SiOI_Density']).to_ndarray()
    return arr

@derived_field(name='GHCNI_ndensity', sampling_type='cell')
def GHCNI_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 27.0
    arr = (num_unit*data['GHCNI_Density']).to_ndarray()
    return arr

@derived_field(name='GHNCI_ndensity', sampling_type='cell')
def GHNCI_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 27.0
    arr = (num_unit*data['GHNCI_Density']).to_ndarray()
    return arr

@derived_field(name='GHNCOI_ndensity', sampling_type='cell')
def GHNCOI_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 43.0
    arr = (num_unit*data['GHNCOI_Density']).to_ndarray()
    return arr

@derived_field(name='GHNOI_ndensity', sampling_type='cell')
def GHNOI_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 31.0
    arr = (num_unit*data['GHNOI_Density']).to_ndarray()
    return arr

@derived_field(name='GMgI_ndensity', sampling_type='cell')
def GMgI_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 24.0
    arr = (num_unit*data['GMgI_Density']).to_ndarray()
    return arr

@derived_field(name='GN2I_ndensity', sampling_type='cell')
def GN2I_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 28.0
    arr = (num_unit*data['GN2I_Density']).to_ndarray()
    return arr

@derived_field(name='GNH3I_ndensity', sampling_type='cell')
def GNH3I_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 17.0
    arr = (num_unit*data['GNH3I_Density']).to_ndarray()
    return arr

@derived_field(name='GNOI_ndensity', sampling_type='cell')
def GNOI_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 30.0
    arr = (num_unit*data['GNOI_Density']).to_ndarray()
    return arr

@derived_field(name='GNO2I_ndensity', sampling_type='cell')
def GNO2I_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 46.0
    arr = (num_unit*data['GNO2I_Density']).to_ndarray()
    return arr

@derived_field(name='GO2I_ndensity', sampling_type='cell')
def GO2I_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 32.0
    arr = (num_unit*data['GO2I_Density']).to_ndarray()
    return arr

@derived_field(name='GO2HI_ndensity', sampling_type='cell')
def GO2HI_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 33.0
    arr = (num_unit*data['GO2HI_Density']).to_ndarray()
    return arr

@derived_field(name='GSiCI_ndensity', sampling_type='cell')
def GSiCI_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 40.0
    arr = (num_unit*data['GSiCI_Density']).to_ndarray()
    return arr

@derived_field(name='GSiC2I_ndensity', sampling_type='cell')
def GSiC2I_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 52.0
    arr = (num_unit*data['GSiC2I_Density']).to_ndarray()
    return arr

@derived_field(name='GSiC3I_ndensity', sampling_type='cell')
def GSiC3I_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 64.0
    arr = (num_unit*data['GSiC3I_Density']).to_ndarray()
    return arr

@derived_field(name='GSiH4I_ndensity', sampling_type='cell')
def GSiH4I_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 32.0
    arr = (num_unit*data['GSiH4I_Density']).to_ndarray()
    return arr

@derived_field(name='GSiOI_ndensity', sampling_type='cell')
def GSiOI_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 44.0
    arr = (num_unit*data['GSiOI_Density']).to_ndarray()
    return arr

@derived_field(name='CI_ndensity', sampling_type='cell')
def CI_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 12.0
    arr = (num_unit*data['CI_Density']).to_ndarray()
    return arr

@derived_field(name='CII_ndensity', sampling_type='cell')
def CII_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 12.0
    arr = (num_unit*data['CII_Density']).to_ndarray()
    return arr

@derived_field(name='CHI_ndensity', sampling_type='cell')
def CHI_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 13.0
    arr = (num_unit*data['CHI_Density']).to_ndarray()
    return arr

@derived_field(name='CHII_ndensity', sampling_type='cell')
def CHII_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 13.0
    arr = (num_unit*data['CHII_Density']).to_ndarray()
    return arr

@derived_field(name='CH2I_ndensity', sampling_type='cell')
def CH2I_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 14.0
    arr = (num_unit*data['CH2I_Density']).to_ndarray()
    return arr

@derived_field(name='CH2II_ndensity', sampling_type='cell')
def CH2II_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 14.0
    arr = (num_unit*data['CH2II_Density']).to_ndarray()
    return arr

@derived_field(name='CH3I_ndensity', sampling_type='cell')
def CH3I_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 15.0
    arr = (num_unit*data['CH3I_Density']).to_ndarray()
    return arr

@derived_field(name='CH3II_ndensity', sampling_type='cell')
def CH3II_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 15.0
    arr = (num_unit*data['CH3II_Density']).to_ndarray()
    return arr

@derived_field(name='CH3OHI_ndensity', sampling_type='cell')
def CH3OHI_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 32.0
    arr = (num_unit*data['CH3OHI_Density']).to_ndarray()
    return arr

@derived_field(name='CH4I_ndensity', sampling_type='cell')
def CH4I_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 16.0
    arr = (num_unit*data['CH4I_Density']).to_ndarray()
    return arr

@derived_field(name='CH4II_ndensity', sampling_type='cell')
def CH4II_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 16.0
    arr = (num_unit*data['CH4II_Density']).to_ndarray()
    return arr

@derived_field(name='CNI_ndensity', sampling_type='cell')
def CNI_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 26.0
    arr = (num_unit*data['CNI_Density']).to_ndarray()
    return arr

@derived_field(name='CNII_ndensity', sampling_type='cell')
def CNII_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 26.0
    arr = (num_unit*data['CNII_Density']).to_ndarray()
    return arr

@derived_field(name='COI_ndensity', sampling_type='cell')
def COI_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 28.0
    arr = (num_unit*data['COI_Density']).to_ndarray()
    return arr

@derived_field(name='COII_ndensity', sampling_type='cell')
def COII_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 28.0
    arr = (num_unit*data['COII_Density']).to_ndarray()
    return arr

@derived_field(name='CO2I_ndensity', sampling_type='cell')
def CO2I_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 44.0
    arr = (num_unit*data['CO2I_Density']).to_ndarray()
    return arr

@derived_field(name='Electron_ndensity', sampling_type='cell')
def Electron_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 1.0
    arr = (num_unit*data['Electron_Density']).to_ndarray()
    return arr

@derived_field(name='HI_ndensity', sampling_type='cell')
def HI_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 1.0
    arr = (num_unit*data['HI_Density']).to_ndarray()
    return arr

@derived_field(name='HII_ndensity', sampling_type='cell')
def HII_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 1.0
    arr = (num_unit*data['HII_Density']).to_ndarray()
    return arr

@derived_field(name='H2I_ndensity', sampling_type='cell')
def H2I_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 2.0
    arr = (num_unit*data['H2I_Density']).to_ndarray()
    return arr

@derived_field(name='H2II_ndensity', sampling_type='cell')
def H2II_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 2.0
    arr = (num_unit*data['H2II_Density']).to_ndarray()
    return arr

@derived_field(name='H2CNI_ndensity', sampling_type='cell')
def H2CNI_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 28.0
    arr = (num_unit*data['H2CNI_Density']).to_ndarray()
    return arr

@derived_field(name='H2COI_ndensity', sampling_type='cell')
def H2COI_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 30.0
    arr = (num_unit*data['H2COI_Density']).to_ndarray()
    return arr

@derived_field(name='H2COII_ndensity', sampling_type='cell')
def H2COII_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 30.0
    arr = (num_unit*data['H2COII_Density']).to_ndarray()
    return arr

@derived_field(name='H2NOII_ndensity', sampling_type='cell')
def H2NOII_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 32.0
    arr = (num_unit*data['H2NOII_Density']).to_ndarray()
    return arr

@derived_field(name='H2OI_ndensity', sampling_type='cell')
def H2OI_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 18.0
    arr = (num_unit*data['H2OI_Density']).to_ndarray()
    return arr

@derived_field(name='H2OII_ndensity', sampling_type='cell')
def H2OII_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 18.0
    arr = (num_unit*data['H2OII_Density']).to_ndarray()
    return arr

@derived_field(name='H2SiOI_ndensity', sampling_type='cell')
def H2SiOI_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 46.0
    arr = (num_unit*data['H2SiOI_Density']).to_ndarray()
    return arr

@derived_field(name='H3II_ndensity', sampling_type='cell')
def H3II_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 3.0
    arr = (num_unit*data['H3II_Density']).to_ndarray()
    return arr

@derived_field(name='H3COII_ndensity', sampling_type='cell')
def H3COII_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 31.0
    arr = (num_unit*data['H3COII_Density']).to_ndarray()
    return arr

@derived_field(name='H3OII_ndensity', sampling_type='cell')
def H3OII_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 19.0
    arr = (num_unit*data['H3OII_Density']).to_ndarray()
    return arr

@derived_field(name='HCNI_ndensity', sampling_type='cell')
def HCNI_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 27.0
    arr = (num_unit*data['HCNI_Density']).to_ndarray()
    return arr

@derived_field(name='HCNII_ndensity', sampling_type='cell')
def HCNII_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 27.0
    arr = (num_unit*data['HCNII_Density']).to_ndarray()
    return arr

@derived_field(name='HCNHII_ndensity', sampling_type='cell')
def HCNHII_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 28.0
    arr = (num_unit*data['HCNHII_Density']).to_ndarray()
    return arr

@derived_field(name='HCOI_ndensity', sampling_type='cell')
def HCOI_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 29.0
    arr = (num_unit*data['HCOI_Density']).to_ndarray()
    return arr

@derived_field(name='HCOII_ndensity', sampling_type='cell')
def HCOII_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 29.0
    arr = (num_unit*data['HCOII_Density']).to_ndarray()
    return arr

@derived_field(name='HCO2II_ndensity', sampling_type='cell')
def HCO2II_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 45.0
    arr = (num_unit*data['HCO2II_Density']).to_ndarray()
    return arr

@derived_field(name='HeI_ndensity', sampling_type='cell')
def HeI_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 4.0
    arr = (num_unit*data['HeI_Density']).to_ndarray()
    return arr

@derived_field(name='HeII_ndensity', sampling_type='cell')
def HeII_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 4.0
    arr = (num_unit*data['HeII_Density']).to_ndarray()
    return arr

@derived_field(name='HeHII_ndensity', sampling_type='cell')
def HeHII_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 5.0
    arr = (num_unit*data['HeHII_Density']).to_ndarray()
    return arr

@derived_field(name='HNCI_ndensity', sampling_type='cell')
def HNCI_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 27.0
    arr = (num_unit*data['HNCI_Density']).to_ndarray()
    return arr

@derived_field(name='HNCOI_ndensity', sampling_type='cell')
def HNCOI_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 43.0
    arr = (num_unit*data['HNCOI_Density']).to_ndarray()
    return arr

@derived_field(name='HNOI_ndensity', sampling_type='cell')
def HNOI_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 31.0
    arr = (num_unit*data['HNOI_Density']).to_ndarray()
    return arr

@derived_field(name='HNOII_ndensity', sampling_type='cell')
def HNOII_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 31.0
    arr = (num_unit*data['HNOII_Density']).to_ndarray()
    return arr

@derived_field(name='HOCII_ndensity', sampling_type='cell')
def HOCII_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 29.0
    arr = (num_unit*data['HOCII_Density']).to_ndarray()
    return arr

@derived_field(name='MgI_ndensity', sampling_type='cell')
def MgI_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 24.0
    arr = (num_unit*data['MgI_Density']).to_ndarray()
    return arr

@derived_field(name='MgII_ndensity', sampling_type='cell')
def MgII_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 24.0
    arr = (num_unit*data['MgII_Density']).to_ndarray()
    return arr

@derived_field(name='NI_ndensity', sampling_type='cell')
def NI_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 14.0
    arr = (num_unit*data['NI_Density']).to_ndarray()
    return arr

@derived_field(name='NII_ndensity', sampling_type='cell')
def NII_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 14.0
    arr = (num_unit*data['NII_Density']).to_ndarray()
    return arr

@derived_field(name='N2I_ndensity', sampling_type='cell')
def N2I_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 28.0
    arr = (num_unit*data['N2I_Density']).to_ndarray()
    return arr

@derived_field(name='N2II_ndensity', sampling_type='cell')
def N2II_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 28.0
    arr = (num_unit*data['N2II_Density']).to_ndarray()
    return arr

@derived_field(name='N2HII_ndensity', sampling_type='cell')
def N2HII_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 29.0
    arr = (num_unit*data['N2HII_Density']).to_ndarray()
    return arr

@derived_field(name='NHI_ndensity', sampling_type='cell')
def NHI_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 15.0
    arr = (num_unit*data['NHI_Density']).to_ndarray()
    return arr

@derived_field(name='NHII_ndensity', sampling_type='cell')
def NHII_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 15.0
    arr = (num_unit*data['NHII_Density']).to_ndarray()
    return arr

@derived_field(name='NH2I_ndensity', sampling_type='cell')
def NH2I_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 16.0
    arr = (num_unit*data['NH2I_Density']).to_ndarray()
    return arr

@derived_field(name='NH2II_ndensity', sampling_type='cell')
def NH2II_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 16.0
    arr = (num_unit*data['NH2II_Density']).to_ndarray()
    return arr

@derived_field(name='NH3I_ndensity', sampling_type='cell')
def NH3I_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 17.0
    arr = (num_unit*data['NH3I_Density']).to_ndarray()
    return arr

@derived_field(name='NH3II_ndensity', sampling_type='cell')
def NH3II_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 17.0
    arr = (num_unit*data['NH3II_Density']).to_ndarray()
    return arr

@derived_field(name='NOI_ndensity', sampling_type='cell')
def NOI_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 30.0
    arr = (num_unit*data['NOI_Density']).to_ndarray()
    return arr

@derived_field(name='NOII_ndensity', sampling_type='cell')
def NOII_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 30.0
    arr = (num_unit*data['NOII_Density']).to_ndarray()
    return arr

@derived_field(name='NO2I_ndensity', sampling_type='cell')
def NO2I_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 46.0
    arr = (num_unit*data['NO2I_Density']).to_ndarray()
    return arr

@derived_field(name='OI_ndensity', sampling_type='cell')
def OI_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 16.0
    arr = (num_unit*data['OI_Density']).to_ndarray()
    return arr

@derived_field(name='OII_ndensity', sampling_type='cell')
def OII_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 16.0
    arr = (num_unit*data['OII_Density']).to_ndarray()
    return arr

@derived_field(name='O2I_ndensity', sampling_type='cell')
def O2I_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 32.0
    arr = (num_unit*data['O2I_Density']).to_ndarray()
    return arr

@derived_field(name='O2II_ndensity', sampling_type='cell')
def O2II_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 32.0
    arr = (num_unit*data['O2II_Density']).to_ndarray()
    return arr

@derived_field(name='O2HI_ndensity', sampling_type='cell')
def O2HI_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 33.0
    arr = (num_unit*data['O2HI_Density']).to_ndarray()
    return arr

@derived_field(name='O2HII_ndensity', sampling_type='cell')
def O2HII_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 33.0
    arr = (num_unit*data['O2HII_Density']).to_ndarray()
    return arr

@derived_field(name='OCNI_ndensity', sampling_type='cell')
def OCNI_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 42.0
    arr = (num_unit*data['OCNI_Density']).to_ndarray()
    return arr

@derived_field(name='OHI_ndensity', sampling_type='cell')
def OHI_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 17.0
    arr = (num_unit*data['OHI_Density']).to_ndarray()
    return arr

@derived_field(name='OHII_ndensity', sampling_type='cell')
def OHII_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 17.0
    arr = (num_unit*data['OHII_Density']).to_ndarray()
    return arr

@derived_field(name='SiI_ndensity', sampling_type='cell')
def SiI_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 28.0
    arr = (num_unit*data['SiI_Density']).to_ndarray()
    return arr

@derived_field(name='SiII_ndensity', sampling_type='cell')
def SiII_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 28.0
    arr = (num_unit*data['SiII_Density']).to_ndarray()
    return arr

@derived_field(name='SiCI_ndensity', sampling_type='cell')
def SiCI_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 40.0
    arr = (num_unit*data['SiCI_Density']).to_ndarray()
    return arr

@derived_field(name='SiCII_ndensity', sampling_type='cell')
def SiCII_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 40.0
    arr = (num_unit*data['SiCII_Density']).to_ndarray()
    return arr

@derived_field(name='SiC2I_ndensity', sampling_type='cell')
def SiC2I_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 52.0
    arr = (num_unit*data['SiC2I_Density']).to_ndarray()
    return arr

@derived_field(name='SiC2II_ndensity', sampling_type='cell')
def SiC2II_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 52.0
    arr = (num_unit*data['SiC2II_Density']).to_ndarray()
    return arr

@derived_field(name='SiC3I_ndensity', sampling_type='cell')
def SiC3I_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 64.0
    arr = (num_unit*data['SiC3I_Density']).to_ndarray()
    return arr

@derived_field(name='SiC3II_ndensity', sampling_type='cell')
def SiC3II_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 64.0
    arr = (num_unit*data['SiC3II_Density']).to_ndarray()
    return arr

@derived_field(name='SiHI_ndensity', sampling_type='cell')
def SiHI_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 29.0
    arr = (num_unit*data['SiHI_Density']).to_ndarray()
    return arr

@derived_field(name='SiHII_ndensity', sampling_type='cell')
def SiHII_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 29.0
    arr = (num_unit*data['SiHII_Density']).to_ndarray()
    return arr

@derived_field(name='SiH2I_ndensity', sampling_type='cell')
def SiH2I_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 30.0
    arr = (num_unit*data['SiH2I_Density']).to_ndarray()
    return arr

@derived_field(name='SiH2II_ndensity', sampling_type='cell')
def SiH2II_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 30.0
    arr = (num_unit*data['SiH2II_Density']).to_ndarray()
    return arr

@derived_field(name='SiH3I_ndensity', sampling_type='cell')
def SiH3I_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 31.0
    arr = (num_unit*data['SiH3I_Density']).to_ndarray()
    return arr

@derived_field(name='SiH3II_ndensity', sampling_type='cell')
def SiH3II_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 31.0
    arr = (num_unit*data['SiH3II_Density']).to_ndarray()
    return arr

@derived_field(name='SiH4I_ndensity', sampling_type='cell')
def SiH4I_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 32.0
    arr = (num_unit*data['SiH4I_Density']).to_ndarray()
    return arr

@derived_field(name='SiH4II_ndensity', sampling_type='cell')
def SiH4II_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 32.0
    arr = (num_unit*data['SiH4II_Density']).to_ndarray()
    return arr

@derived_field(name='SiH5II_ndensity', sampling_type='cell')
def SiH5II_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 33.0
    arr = (num_unit*data['SiH5II_Density']).to_ndarray()
    return arr

@derived_field(name='SiOI_ndensity', sampling_type='cell')
def SiOI_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 44.0
    arr = (num_unit*data['SiOI_Density']).to_ndarray()
    return arr

@derived_field(name='SiOII_ndensity', sampling_type='cell')
def SiOII_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 44.0
    arr = (num_unit*data['SiOII_Density']).to_ndarray()
    return arr

@derived_field(name='SiOHII_ndensity', sampling_type='cell')
def SiOHII_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    dunit = data.ds.mass_unit/data.ds.length_unit**3
    num_unit = dunit / mh_cgs / 45.0
    arr = (num_unit*data['SiOHII_Density']).to_ndarray()
    return arr

@derived_field(name='element_E_ndensity', sampling_type='cell')
def element_E_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    arr = (0.0 + 1*data['Electron_ndensity'])
    return arr

@derived_field(name='element_H_ndensity', sampling_type='cell')
def element_H_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    arr = (0.0 + 4*data['GCH3OHI_ndensity'] + 4*data['GCH4I_ndensity']
          + 2*data['GH2CNI_ndensity'] + 2*data['GH2COI_ndensity'] +
          2*data['GH2OI_ndensity'] + 2*data['GH2SiOI_ndensity'] +
          1*data['GHCNI_ndensity'] + 1*data['GHNCI_ndensity'] +
          1*data['GHNCOI_ndensity'] + 1*data['GHNOI_ndensity'] +
          3*data['GNH3I_ndensity'] + 1*data['GO2HI_ndensity'] +
          4*data['GSiH4I_ndensity'] + 1*data['CHI_ndensity'] +
          1*data['CHII_ndensity'] + 2*data['CH2I_ndensity'] +
          2*data['CH2II_ndensity'] + 3*data['CH3I_ndensity'] +
          3*data['CH3II_ndensity'] + 4*data['CH3OHI_ndensity'] +
          4*data['CH4I_ndensity'] + 4*data['CH4II_ndensity'] +
          1*data['HI_ndensity'] + 1*data['HII_ndensity'] +
          2*data['H2I_ndensity'] + 2*data['H2II_ndensity'] +
          2*data['H2CNI_ndensity'] + 2*data['H2COI_ndensity'] +
          2*data['H2COII_ndensity'] + 2*data['H2NOII_ndensity'] +
          2*data['H2OI_ndensity'] + 2*data['H2OII_ndensity'] +
          2*data['H2SiOI_ndensity'] + 3*data['H3II_ndensity'] +
          3*data['H3COII_ndensity'] + 3*data['H3OII_ndensity'] +
          1*data['HCNI_ndensity'] + 1*data['HCNII_ndensity'] +
          2*data['HCNHII_ndensity'] + 1*data['HCOI_ndensity'] +
          1*data['HCOII_ndensity'] + 1*data['HCO2II_ndensity'] +
          1*data['HeHII_ndensity'] + 1*data['HNCI_ndensity'] +
          1*data['HNCOI_ndensity'] + 1*data['HNOI_ndensity'] +
          1*data['HNOII_ndensity'] + 1*data['HOCII_ndensity'] +
          1*data['N2HII_ndensity'] + 1*data['NHI_ndensity'] +
          1*data['NHII_ndensity'] + 2*data['NH2I_ndensity'] +
          2*data['NH2II_ndensity'] + 3*data['NH3I_ndensity'] +
          3*data['NH3II_ndensity'] + 1*data['O2HI_ndensity'] +
          1*data['O2HII_ndensity'] + 1*data['OHI_ndensity'] +
          1*data['OHII_ndensity'] + 1*data['SiHI_ndensity'] +
          1*data['SiHII_ndensity'] + 2*data['SiH2I_ndensity'] +
          2*data['SiH2II_ndensity'] + 3*data['SiH3I_ndensity'] +
          3*data['SiH3II_ndensity'] + 4*data['SiH4I_ndensity'] +
          4*data['SiH4II_ndensity'] + 5*data['SiH5II_ndensity'] +
          1*data['SiOHII_ndensity'])
    return arr

@derived_field(name='element_D_ndensity', sampling_type='cell')
def element_D_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    arr = (0.0)
    return arr

@derived_field(name='element_HE_ndensity', sampling_type='cell')
def element_HE_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    arr = (0.0 + 1*data['HeI_ndensity'] + 1*data['HeII_ndensity'] +
          1*data['HeHII_ndensity'])
    return arr

@derived_field(name='element_C_ndensity', sampling_type='cell')
def element_C_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    arr = (0.0 + 1*data['GCH3OHI_ndensity'] + 1*data['GCH4I_ndensity']
          + 1*data['GCOI_ndensity'] + 1*data['GCO2I_ndensity'] +
          1*data['GH2CNI_ndensity'] + 1*data['GH2COI_ndensity'] +
          1*data['GHCNI_ndensity'] + 1*data['GHNCI_ndensity'] +
          1*data['GHNCOI_ndensity'] + 1*data['GSiCI_ndensity'] +
          2*data['GSiC2I_ndensity'] + 3*data['GSiC3I_ndensity'] +
          1*data['CI_ndensity'] + 1*data['CII_ndensity'] +
          1*data['CHI_ndensity'] + 1*data['CHII_ndensity'] +
          1*data['CH2I_ndensity'] + 1*data['CH2II_ndensity'] +
          1*data['CH3I_ndensity'] + 1*data['CH3II_ndensity'] +
          1*data['CH3OHI_ndensity'] + 1*data['CH4I_ndensity'] +
          1*data['CH4II_ndensity'] + 1*data['CNI_ndensity'] +
          1*data['CNII_ndensity'] + 1*data['COI_ndensity'] +
          1*data['COII_ndensity'] + 1*data['CO2I_ndensity'] +
          1*data['H2CNI_ndensity'] + 1*data['H2COI_ndensity'] +
          1*data['H2COII_ndensity'] + 1*data['H3COII_ndensity'] +
          1*data['HCNI_ndensity'] + 1*data['HCNII_ndensity'] +
          1*data['HCNHII_ndensity'] + 1*data['HCOI_ndensity'] +
          1*data['HCOII_ndensity'] + 1*data['HCO2II_ndensity'] +
          1*data['HNCI_ndensity'] + 1*data['HNCOI_ndensity'] +
          1*data['HOCII_ndensity'] + 1*data['OCNI_ndensity'] +
          1*data['SiCI_ndensity'] + 1*data['SiCII_ndensity'] +
          2*data['SiC2I_ndensity'] + 2*data['SiC2II_ndensity'] +
          3*data['SiC3I_ndensity'] + 3*data['SiC3II_ndensity'])
    return arr

@derived_field(name='element_N_ndensity', sampling_type='cell')
def element_N_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    arr = (0.0 + 1*data['GH2CNI_ndensity'] + 1*data['GHCNI_ndensity'] +
          1*data['GHNCI_ndensity'] + 1*data['GHNCOI_ndensity'] +
          1*data['GHNOI_ndensity'] + 2*data['GN2I_ndensity'] +
          1*data['GNH3I_ndensity'] + 1*data['GNOI_ndensity'] +
          1*data['GNO2I_ndensity'] + 1*data['CNI_ndensity'] +
          1*data['CNII_ndensity'] + 1*data['H2CNI_ndensity'] +
          1*data['H2NOII_ndensity'] + 1*data['HCNI_ndensity'] +
          1*data['HCNII_ndensity'] + 1*data['HCNHII_ndensity'] +
          1*data['HNCI_ndensity'] + 1*data['HNCOI_ndensity'] +
          1*data['HNOI_ndensity'] + 1*data['HNOII_ndensity'] +
          1*data['NI_ndensity'] + 1*data['NII_ndensity'] +
          2*data['N2I_ndensity'] + 2*data['N2II_ndensity'] +
          2*data['N2HII_ndensity'] + 1*data['NHI_ndensity'] +
          1*data['NHII_ndensity'] + 1*data['NH2I_ndensity'] +
          1*data['NH2II_ndensity'] + 1*data['NH3I_ndensity'] +
          1*data['NH3II_ndensity'] + 1*data['NOI_ndensity'] +
          1*data['NOII_ndensity'] + 1*data['NO2I_ndensity'] +
          1*data['OCNI_ndensity'])
    return arr

@derived_field(name='element_O_ndensity', sampling_type='cell')
def element_O_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    arr = (0.0 + 1*data['GCH3OHI_ndensity'] + 1*data['GCOI_ndensity'] +
          2*data['GCO2I_ndensity'] + 1*data['GH2COI_ndensity'] +
          1*data['GH2OI_ndensity'] + 1*data['GH2SiOI_ndensity'] +
          1*data['GHNCOI_ndensity'] + 1*data['GHNOI_ndensity'] +
          1*data['GNOI_ndensity'] + 2*data['GNO2I_ndensity'] +
          2*data['GO2I_ndensity'] + 2*data['GO2HI_ndensity'] +
          1*data['GSiOI_ndensity'] + 1*data['CH3OHI_ndensity'] +
          1*data['COI_ndensity'] + 1*data['COII_ndensity'] +
          2*data['CO2I_ndensity'] + 1*data['H2COI_ndensity'] +
          1*data['H2COII_ndensity'] + 1*data['H2NOII_ndensity'] +
          1*data['H2OI_ndensity'] + 1*data['H2OII_ndensity'] +
          1*data['H2SiOI_ndensity'] + 1*data['H3COII_ndensity'] +
          1*data['H3OII_ndensity'] + 1*data['HCOI_ndensity'] +
          1*data['HCOII_ndensity'] + 2*data['HCO2II_ndensity'] +
          1*data['HNCOI_ndensity'] + 1*data['HNOI_ndensity'] +
          1*data['HNOII_ndensity'] + 1*data['HOCII_ndensity'] +
          1*data['NOI_ndensity'] + 1*data['NOII_ndensity'] +
          2*data['NO2I_ndensity'] + 1*data['OI_ndensity'] +
          1*data['OII_ndensity'] + 2*data['O2I_ndensity'] +
          2*data['O2II_ndensity'] + 2*data['O2HI_ndensity'] +
          2*data['O2HII_ndensity'] + 1*data['OCNI_ndensity'] +
          1*data['OHI_ndensity'] + 1*data['OHII_ndensity'] +
          1*data['SiOI_ndensity'] + 1*data['SiOII_ndensity'] +
          1*data['SiOHII_ndensity'])
    return arr

@derived_field(name='element_MG_ndensity', sampling_type='cell')
def element_MG_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    arr = (0.0 + 1*data['GMgI_ndensity'] + 1*data['MgI_ndensity'] +
          1*data['MgII_ndensity'])
    return arr

@derived_field(name='element_SI_ndensity', sampling_type='cell')
def element_SI_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    arr = (0.0 + 1*data['GH2SiOI_ndensity'] + 1*data['GSiCI_ndensity']
          + 1*data['GSiC2I_ndensity'] + 1*data['GSiC3I_ndensity'] +
          1*data['GSiH4I_ndensity'] + 1*data['GSiOI_ndensity'] +
          1*data['H2SiOI_ndensity'] + 1*data['SiI_ndensity'] +
          1*data['SiII_ndensity'] + 1*data['SiCI_ndensity'] +
          1*data['SiCII_ndensity'] + 1*data['SiC2I_ndensity'] +
          1*data['SiC2II_ndensity'] + 1*data['SiC3I_ndensity'] +
          1*data['SiC3II_ndensity'] + 1*data['SiHI_ndensity'] +
          1*data['SiHII_ndensity'] + 1*data['SiH2I_ndensity'] +
          1*data['SiH2II_ndensity'] + 1*data['SiH3I_ndensity'] +
          1*data['SiH3II_ndensity'] + 1*data['SiH4I_ndensity'] +
          1*data['SiH4II_ndensity'] + 1*data['SiH5II_ndensity'] +
          1*data['SiOI_ndensity'] + 1*data['SiOII_ndensity'] +
          1*data['SiOHII_ndensity'])
    return arr

@derived_field(name='element_S_ndensity', sampling_type='cell')
def element_S_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    arr = (0.0)
    return arr

@derived_field(name='element_CL_ndensity', sampling_type='cell')
def element_CL_ndensity(field, data):
    if 'enzo' not in data.ds.dataset_type:
        return
    if data.ds.parameters['MultiSpecies'] < 4:
        return
    arr = (0.0)
    return arr