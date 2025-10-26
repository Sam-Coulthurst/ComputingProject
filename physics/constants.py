from astropy import units as u

G_SI    = 6.6726e-11 * u.m**3 * u.kg**-1 * u.s**-2
m_earth_SI = 5.9742e24 * u.kg
m_moon_SI = 7.35e22 * u.kg
d_earth_moon_SI = 3.844e8 * u.m
print(type(d_earth_moon_SI))
# Calculate canonical units
M_tot = (m_earth_SI + m_moon_SI)
L_unit = d_earth_moon_SI              # canonical length unit
T_unit = ((L_unit**3 / (G_SI * M_tot))**0.5).to(u.s)  # time unit
V_unit = (L_unit / T_unit).to(u.m / u.s)           # velocity unit
A_unit = (L_unit / T_unit**2).to(u.m / u.s**2)     # acceleration unit

# Dimensionless constants
mu = (m_moon_SI / M_tot).decompose().value     # mass ratio m2 / (m1+m2)
m1_dimless = 1.0 - mu
m2_dimless = mu
G_dimless = 1.0   

# Export constants
L_UNIT = L_unit.value      # in meters
T_UNIT = T_unit.value      # in seconds
V_UNIT = V_unit.value      # in m/s
A_UNIT = A_unit.value      # in m/s^2
MU = mu
M1 = m1_dimless
M2 = m2_dimless
G_DIMLESS = G_dimless

