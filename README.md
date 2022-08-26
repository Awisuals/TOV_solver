# TOV_solver

# Program to solve the astrophysical hydrostatic equilibrium equation - Tolman-Oppenheimer-Volkoff equation as well as to model some astrophysical bodies.

Tämän ohjelma ratkaisee yhdistetyn epälineaarisen
differentiaaliryhmän. Yhtälöt ovat ns. TOV-yhtälöt, jotka kuvaavat yhdessä
tähden rakenteen.

Ohjelmalla voi määrittää jollekkin tähtityypille massa-säde - relaation
mallintamalla useaa tähteä annetuilla parametreilla sekä joillakin
tilanyhtälöillä ja varioimalla esim. tähden keskipisteen energiatiheyttä.

Notaatio // Notation:
    WD = White Dwarf, NS = Neutron Star

Valitaan geometrisoidut yksiköt // Choose geometrized units:
    G = c = 1

Yhtälöt // Equations:

    dmdr = 4*np.pi*rho*r**2
    dpdr = -(rho+p)*(m + 4*np.pi*r**3*p)/(r*(r-2*m))   # Relativistinen
    dpdr = -(m*rho)/(r**2)                             # Ei-Relativistinen

Näiden lisäksi tarvitaan tilanyhtälö. Valkoiselle kääpiölle valitaan
paineen ja energiatiheyden relatoiva polytrooppimalli:

    p = Kappa*rho**Gamma

Neutronitähdelle valitaan sopiva(t) malli(t) paperista
"A unified equation of state of dense matter and neutron star
structure".:

    Taulukot // tables 3. ja 5.

Ohjelmalla voi lakea avaruuden kaarevuusskalaarin (R) - Riccin skalaari -
ja piirtää tästä kuvaajan.:

    R = -8*np.pi*G*(rho - 3*p)
    
// Eng.

This program solves the combined nonlinear DE-equation group. 
The equations are the so-called TOV equations that describe together
star structure.

The program can be used to define a mass-radius relation for a star type
by modeling several stars with the given parameters as well as some
with equations of state and by varying, for example, the energy density of
the center of the star.

Notaatio // Notation:
    WD = White Dwarf, NS = Neutron Star

Valitaan geometrisoidut yksiköt // Choose geometrized units:
    G = c = 1

Yhtälöt // Equations:

    dmdr = 4*np.pi*rho*r**2
    dpdr = -(rho+p)*(m + 4*np.pi*r**3*p)/(r*(r-2*m))   # Relativistinen
    dpdr = -(m*rho)/(r**2)                             # Ei-Relativistinen
    
In addition to these, an equation of state is needed. A white dwarf is chosen
polytropic model relating pressure and energy density:

    p = Kappa*rho**Gamma
    
Suitable model(s) are selected from the paper for the neutron star
"A unified equation of state of dense matter and Neutron star
structure".:

    Taulukot // tables 3. ja 5.
    
The program can be used to calculate the curvature scalar (R) of space 
- Ricci's scalar - and draw a graph of this.:

    R = -8*np.pi*G*(rho - 3*p)

Esimerkkejä koodin ajamisesta konsolissa
//
Examples of running code in the console:
    TODO WD - rakenne
    TODO WD Massa-säde - relaatio
    TODO WD avaruuden kaarevuus
    TODO NS - rakenne
    TODO NS avaruuden kaarevuus
    TODO kivieksoplaneetta

Parametrien arvojen kokoluokka
//
Size range of parameter values:

Valkoinen kääpiö // White dwarf (geom. units):
    R0 = ~R_earth = 6e6
    Kappa = ~ 0.2-20
    rho_c = ~1e-10

Neutronitähti // Neutron star (geom. units):
    R0 = ~10km = 10000m
    Kappa = ~20-1000

REL (TOV):
    n = 3
    Gamma = 4/3

NOT-REL (NEWT.):
    n = 1.5
    Gamma = 5/3
