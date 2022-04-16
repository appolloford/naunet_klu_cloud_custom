from __future__ import annotations
from enum import IntEnum
from naunet.network import define_reaction, define_dust
from naunet.reactions.reaction import Reaction, ReactionType as BasicType
from naunet.dusts.dust import Dust
from naunet.dusts.rr07dust import RR07Dust
from naunet.species import Species



@define_reaction(name="uclcustreaction")
class CUSTOMReaction(Reaction):
    """
    The reaction format of UCLCHEM Makerates output.
    """

    class ReactionType(IntEnum):
        # two-body gas-phase reaction
        UCLCHEM_MA = BasicType.GAS_TWOBODY
        # direct cosmic-ray ionisation
        UCLCHEM_CR = BasicType.GAS_COSMICRAY
        # cosmic-ray-induced photoreaction
        UCLCHEM_CP = BasicType.GAS_UMIST_CRPHOT
        # photoreaction
        UCLCHEM_PH = BasicType.GAS_PHOTON
        # accretion
        UCLCHEM_FR = BasicType.GRAIN_FREEZE
        # thermal desorption
        UCLCHEM_TH = BasicType.GRAIN_DESORPT_THERMAL
        # cosmic-ray-induced thermal desorption
        UCLCHEM_CD = BasicType.GRAIN_DESORPT_COSMICRAY
        # photodesorption
        UCLCHEM_PD = BasicType.GRAIN_DESORPT_PHOTON
        # reactive desorption
        UCLCHEM_RD = BasicType.GRAIN_DESORPT_REACTIVE
        # H2-formation-induced desorption
        UCLCHEM_HD = BasicType.GRAIN_DESORPT_H2
        # surface diffusion
        UCLCHEM_DF = BasicType.SURFACE_DIFFUSION

    reactant2type = {
        "CRP": ReactionType.UCLCHEM_CR,
        "PHOTON": ReactionType.UCLCHEM_PH,
        "CRPHOT": ReactionType.UCLCHEM_CP,
        "FREEZE": ReactionType.UCLCHEM_FR,
        "DESOH2": ReactionType.UCLCHEM_HD,
        "DESCR": ReactionType.UCLCHEM_CD,
        "DEUVCR": ReactionType.UCLCHEM_PD,
        "THERM": ReactionType.UCLCHEM_TH,
        "DIFF": ReactionType.UCLCHEM_DF,
        "CHEMDES": ReactionType.UCLCHEM_RD,
    }

    consts = {
        "zism": 1.3e-17,
    }

    varis = {
        "nH": None,
        "Tgas": None,
        "zeta": 1.3e-17,
        "Av": 1.0,
        "omega": 0.5,
        "G0": 1.0,
        "uvcreff": 1.0e-3,  # UVCREFF is ratio of CR induced UV to ISRF UV
    }

    locvars = [
        "double h2col = 0.5*1.59e21*Av",
        "double cocol = 1e-5 * h2col",
        "double lamdabar = GetCharactWavelength(h2col, cocol)",
        "double H2shielding = GetShieldingFactor(IDX_H2I, h2col, h2col, Tgas, 1)",
        "double H2formation = 1.0e-17 * sqrt(Tgas) * GetHNuclei(y)",
        "double H2dissociation = 5.1e-11 * G0 * GetGrainScattering(Av, 1000.0) * H2shielding",
    ]

    def __init__(self, react_string) -> None:
        super().__init__(format="uclcust", react_string=react_string)

    def rateexpr(self, dust: Dust = None):
        a = self.alpha
        b = self.beta
        c = self.gamma
        rtype = self.reaction_type

        zeta = f"(zeta / zism)"  # uclchem has cosmic-ray ionization rate in unit of 1.3e-17s-1

        re1 = self.reactants[0]
        # re2 = self.reactants[1] if len(self.reactants) > 1 else None

        # two-body gas-phase reaction
        if rtype == self.ReactionType.UCLCHEM_MA:
            rate = f"{a} * pow(Tgas/300.0, {b}) * exp(-{c}/Tgas)"

        # direct cosmic-ray ionisation
        elif rtype == self.ReactionType.UCLCHEM_CR:
            rate = f"{a} * {zeta}"

        # cosmic-ray-induced photoreaction
        elif rtype == self.ReactionType.UCLCHEM_CP:
            rate = f"{a} * {zeta} * pow(Tgas/300.0, {b}) * {c} / (1.0 - omega)"

        # photoreaction
        elif rtype == self.ReactionType.UCLCHEM_PH:
            rate = f"G0 * {a} * exp(-{c}*Av) / 1.7"  # convert habing to Draine
            if re1.name in ["CO"]:
                shield = f"GetShieldingFactor(IDX_{re1.alias}, h2col, {re1.name.lower()}col, Tgas, 1)"
                rate = f"(2.0e-10) * G0 * {shield} * GetGrainScattering(Av, lamdabar) / 1.7"

        # accretion
        elif rtype == self.ReactionType.UCLCHEM_FR:
            rate = dust.rateexpr(
                self.reaction_type, self.reactants, a, b, c, sym_tgas="Tgas"
            )

        # thermal desorption
        elif rtype == self.ReactionType.UCLCHEM_TH:
            rate = dust.rateexpr(
                self.reaction_type, self.reactants, a, b, c, sym_tdust="Tgas"
            )

        # cosmic-ray-induced thermal desorption
        elif rtype == self.ReactionType.UCLCHEM_CD:
            rate = dust.rateexpr(
                self.reaction_type, self.reactants, a, b, c, sym_cr=zeta
            )

        # photodesorption
        elif rtype == self.ReactionType.UCLCHEM_PD:
            # uvphot = f"({zeta} + (G0/uvcreff) * exp(-1.8*Av) )"
            uvphot = f"G0*1.0e8*exp(-Av*3.02) + 1.0e4 * {zeta}"
            rate = dust.rateexpr(
                self.reaction_type, self.reactants, a, b, c, sym_phot=uvphot
            )

        # H2 formation induced desorption
        elif rtype == self.ReactionType.UCLCHEM_HD:
            # Epsilon is efficieny of this process, number of molecules removed per event
            # h2form is formation rate of h2, dependent on hydrogen abundance.
            h2formrate = f"1.0e-17 * sqrt(Tgas) * y[IDX_HI] * nH"
            rate = dust.rateexpr(
                self.reaction_type, self.reactants, a, b, c, sym_h2form=h2formrate
            )

        else:
            raise ValueError(f"Unsupported type: {rtype}")

        rate = self._beautify(rate)
        return rate

    def _parse_string(self, react_string) -> None:

        kwlist = [
            "NAN",
            "FREEZE",
            "DESOH2",
            "DESCR",
            "DEUVCR",
            "THERM",
            "DIFF",
            "CHEMDES",
        ]

        if react_string.strip() != "":

            *rpspec, a, b, c, lt, ut = react_string.split(",")

            self.reaction_type = self.reactant2type.get(
                rpspec[1], self.ReactionType.UCLCHEM_MA
            )

            # Turn off Freeze-out reaction beyond 30K
            if self.reaction_type == self.ReactionType.UCLCHEM_FR:
                lt, ut = 0, 30

            self.alpha = float(a)
            self.beta = float(b)
            self.gamma = float(c)
            self.temp_min = float(lt)
            self.temp_max = float(ut)
            # UCLCHEM program does not check the temperature range, the next two lines are used for benchmark test
            # self.temp_min = 10.0
            # self.temp_max = 41000.0

            reactants = [r for r in rpspec[0:3] if r not in kwlist]
            products = [p for p in rpspec[3:7] if p not in kwlist]

            self.reactants = [
                self.create_species(r) for r in reactants if self.create_species(r)
            ]
            self.products = [
                self.create_species(p) for p in products if self.create_species(p)
            ]

@define_dust(name="rr07custom")
class CUSTOMDust(RR07Dust):

    consts = {
        "nmono": 2.0,
    }

    varis = {
        "rG": 1e-5,  # grain radius
        "gdens": 7.6394373e-13,  # grain density
        "sites": 1e15,  # surface sites
        "fr": 1.0,  # freeze ratio
        "opt_thd": 1.0,  # thermal desorption option
        "opt_crd": 1.0,  # cosmic ray induced desorption option
        "opt_h2d": 1.0,  # H2 formation induced desorption option
        "opt_uvd": 1.0,  # UV desorption option
        "eb_h2d": 1.21e3,  # maxmium binding energy which H2 desorption can desorb
        "eb_crd": 1.21e3,  # maxmium binding energy which CR desorption can desorb
        "eb_uvd": 1.0e4,  # maxmium binding energy which UV desorption can desorb
        "crdeseff": 1e5,  # cosmic ray desorption efficiency
        "h2deseff": 1.0e-2,  # H2 desorption efficiency
        "ksp": 0.0,  # Sputtering rate
    }

    locvars = [
        # "double mant = GetMantleDens(y) > 0.0 ? GetMantleDens(y) : 1e-40",
        "double mant = GetMantleDens(y)",
        "double mantabund = mant / nH",
        "double garea = (pi*rG*rG) * gdens",  # total grain cross-section
        "double garea_per_H = garea / nH",
        "double densites = 4.0 * garea * sites",
    ]

    def __init__(self, species = None) -> None:
        super().__init__(species=species)
        self.model = "rr07custom"


    def _rate_thermal_desorption(
        self,
        spec: list[Species],
        a: float,
        b: float,
        c: float,
        sym_tdust: str,
    ) -> str:

        if not isinstance(sym_tdust, str):
            raise TypeError("sym_tdust should be a string")

        if not sym_tdust:
            raise ValueError("Symbol of dust temperature does not exist")

        if len(spec) != 1:
            raise ValueError("Number of species in thermal desoprtion should be 1.")

        spec = spec[0]
        rate = " * ".join(
            [
                f"opt_thd",
                f"sqrt(2.0*sites*kerg*eb_{spec.alias}/(pi*pi*amu*{spec.massnumber}))",
                f"2.0 * densites",
                f"exp(-eb_{spec.alias}/{sym_tdust})",
            ],
        )

        rate = f"({rate} + ksp * garea / mant)"
        rate = f"mantabund > 1e-30 ? ({rate}) : 0.0"
        return rate


    def _rate_photon_desorption(
        self,
        spec: list[Species],
        a: float,
        b: float,
        c: float,
        sym_phot: str,
    ) -> str:

        if not isinstance(sym_phot, str):
            raise TypeError("sym_phot should be a string")

        if not sym_phot:
            raise ValueError("Symbol of photon intensity does not exist")

        if len(spec) != 1:
            raise ValueError("Number of species in photon desoprtion should be 1.")

        spec = spec[0]

        rate = f"opt_uvd * ({sym_phot}) * {spec.photon_yield()} * nmono * 4.0 * garea"
        rate = f"mantabund > 1e-30 ? ({rate}) : 0.0"
        return rate

