vpi_0 = 0
vpi_1 = 0
gamma = 3/4

for i in range(0, 5):
    vpi_0_temp = 1 + 3/4 * ( 1/3 * vpi_0 + 2/3 * vpi_1)
    vpi_1 = 2 + 3/4 * ( 2/3 * vpi_0 + 1/3 * vpi_1)
    vpi_0 = vpi_0_temp
    print(f"Iteration = {i:.4f}, vpi_0 = {vpi_0:.4f}, vpi_1 = {vpi_1:.4f}")