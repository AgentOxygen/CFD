# For fitting the timings we used a power function fit through 'scipy.optimize.curve_fit'
alpha, beta = curve_fit(lambda t,a,b: b*(t**a), N, C, p0=(1,0.000001))[0] # We specify initial values to get the algorithm started

# For computing the shear stress and force, we used a second order polynomial fit through 'scipy.optimize.curve_fit'
mm, m, b = curve_fit(lambda t, mm, m, b: mm*(t**2) + m*t + b, y_vals, u_vals)[0]
dudy.append(2*mm*1 + m) # For shear stress, we take the derivative and evaluate at y=1
integral_of_y_dx.append((mm*(1**2) + m*1) - (mm*(0**2) + m*0)) # For force, we take the definite integral over x=[0,1] of shear stress

# To fix the end point at y=1, we can add additional weight to that particular point
# This can be done using the 'sigma' parameter, but can also be done manually by adding a bunch
# of additional points at y=1 with the value of 1 to fit over.
# So for each simulation with different Re
for N, sim in [(40, "re100"), (40, "re200"), (40, "re400"), (40, "re500")]:
    # We parse through and obtain the samples for the lines we created in post processing
    top_x_u, mid_x_u, btm_x_u = getSim(sim, N)
    # Pull them out into a single array of points
    u_vals = np.zeros(3*N)
    u_vals[0:N] = top_x_u[:, 1]
    u_vals[N:2*N] = mid_x_u[:, 1]
    u_vals[2*N:3*N] = btm_x_u[:, 1]

    # Then add a bunch of points at y=1
    n = 10000
    y_vals = np.array([0.1]*n + [0.099]*N + [0.09]*N + [0.08]*N)
    
    du_vals = np.zeros(u_vals.size + n) + 1
    # Then combine the extra "weighting" points with the actual points we want to fit to
    du_vals[n:] = u_vals
    
    # Then use 'curve_fit'