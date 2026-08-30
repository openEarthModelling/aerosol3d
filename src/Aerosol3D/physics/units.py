import pint

# Create a global unit registry for the Aerosol3D project
ureg = pint.UnitRegistry()
Q_ = ureg.Quantity

# Define common units used in aerosol physics to ensure consistency
ureg.define("micrometer = 1e-6 * meter = um = micron")
ureg.define("nanometer = 1e-9 * meter = nm")
