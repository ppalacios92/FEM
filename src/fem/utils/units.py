# units.py

# UNIDADES [SISTEMA BASE SI(mm)]
# DEFINIMOS UN SISTEMA DE UNIDADES CONSISTENTES, SE TRABAJA CON LA UNIDADES:
# =============================================================================
# Utilizamos un sistema de unidades consistentes establecido por:
# LONGITUD -- mm
# FUERZA -- N
# MASA -- tonne (10^3 kg)
# TIEMPO -- s 
# ESFUERZO -- MPa (N/mm2)
# ENERGIA -- mJ (10^-3 J)
# DENSIDAD -- tonne/mm^3
# Transformamos todas las unidades a unidades consistentes 
# =============================================================================

# Length
mm = 1  # [mm]
cm = 10  # [mm]
m = 1000  # [mm]
km = 10**6  # [mm]
inches = 25.4  # [mm]
ft = 304.8  # [mm]
yard = 914.4  # [mm]
mile = 1609344  # [mm]

# Force
N = 1  # [N]
kN = 1000  # [N]
dyne = 1e-5  # [N]
kgf = 9.807  # [N]
tf = 9807  # [N]
lbf = 4.448  # [N]
kip = 4448  # [N]

# Mass
tonne = 1  # [tonne]
kg = 1 / 10**3  # [tonne]
g = 1e-6  # [tonne]
mg = 1e-9  # [tonne]
lb = 453.6 * 10**-6  # [tonne]
oz = 28.3495 * 10**-6  # [tonne]

# Pressure
MPa=1 #[MPa]
kPa=1*10**-3 #[MPa]
GPa=1000 #[MPa]
kgf_cm2=0.09807 #[MPa]
Pa=10**-6 #[MPa]
ksi=6.895 #[MPa]

# Energy
J = 1  # [J]
kJ = 1000  # [J]
mJ = 1e-3  # [J]
cal = 4.184  # [J]
kcal = 4184  # [J]
eV = 1.60218e-19  # [J]
Wh = 3600  # [J]
kWh = 3.6 * 10**6  # [J]

# Power
W = 1  # [W]
kW = 1000  # [W]
MW = 10**6  # [W]
HP = 745.7  # [W]

# Time
s = 1  # [s]
minutes = 60  # [s]
h = 3600  # [s]
day = 24 * h  # [s]
week = 7 * day  # [s]
month = 30 * day  # [s]  # approximate
year = 365.25 * day  # [s]  # average including leap years

# Acceleration
g = 9.81 * mm / s**2

# Angle
radian = 1  # [rad]
degree = (3.141592653589793 / 180)  # [rad]

# Temperature
K = 1  # [K]
C = 1  # [°C]
F = 1  # [°F]

