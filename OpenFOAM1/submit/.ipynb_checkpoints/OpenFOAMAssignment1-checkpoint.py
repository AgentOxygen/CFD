import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as colors
from matplotlib import rc
import numpy as np
from scipy.optimize import curve_fit


width = 20
height = 20
sim_output = np.zeros((6, width, height, 3))

for time_step in range(1, 6):
    # Change to match which simulation to parse
    with open(f"../../sims/o1_cavity/0.{time_step}/U") as f: 
        line = f.readline()
        data_in = False
        index = 0
        while line:
            if line == "(\n":
                data_in = True
            elif line == ")\n":
                data_in = False
            line = f.readline()
            if data_in and len(line.split(" ")) == 3:
                x, y, z = line.split(" ")
                u_x = float(x[1:])
                u_y = float(y)
                u_z = float(z.split(")")[0])
                
                sim_output[time_step][height-int(index / height)-1][index % width] = [u_x, u_y, u_z]
                
                index += 1


# ================================================================ #
# Final Flow Velocity Components ({width}x{height})
# ================================================================ #

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), facecolor='w')
rc('font', **{'size': 15})
f.suptitle(f"Final Flow Velocity Components ({width}x{height})", fontsize=30)
ax1.tick_params(axis='both', which='major', labelsize=15)
ax2.tick_params(axis='both', which='major', labelsize=15)

divider = make_axes_locatable(ax1)
cax1 = divider.append_axes('right', size='5%', pad=0.15)
divider = make_axes_locatable(ax2)
cax2 = divider.append_axes('right', size='5%', pad=0.15)

x = np.linspace(1,0,width)
y = np.linspace(1,0,height)
xx, yy = np.meshgrid(x, y)

vmax = 1
vmin = -0.5
contour1 = ax1.contourf(xx, yy, sim_output[5,:,:,0], cmap='RdBu', norm=colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax), vmin=vmin, vmax=vmax)
f.colorbar(contour1, cax=cax1, orientation='vertical')

contour2 = ax2.contourf(xx, yy, sim_output[5,:,:,1], cmap='RdBu', norm=colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax), vmin=vmin, vmax=vmax)
f.colorbar(contour2, cax=cax2, orientation='vertical')

ax1.set_title("'u' Component", fontsize=20)
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax2.set_title("'v' Component", fontsize=20)
ax2.set_xlabel("x")
ax2.set_ylabel("y")
f.tight_layout()

# ================================================================ #
# Final Flow Velocity Through Center of Cavity (x=0.5)
# ================================================================ #
f, ax1 = plt.subplots(1, 1, figsize=(11, 7), facecolor='w')
rc('font', **{'size': 15})

f.suptitle(f"Final Flow Velocity Through Center of Cavity (x=0.5)", fontsize=30)
ax1.tick_params(axis='both', which='major', labelsize=20)

lw = 4
ax1.plot(y, sim_output[5,:,9,0], color="Blue", label="u", linewidth=lw)
ax1.plot(y, sim_output[5,:,9,1], color="Red", label="v", linewidth=lw)
ax1.set_xlim(0, 1)
ax1.set_ylim(-0.2, 1)
ax1.legend()
ax1.grid()
ax1.set_xlabel("x")
ax1.set_ylabel("y")

f.tight_layout()


# ================================================================ #
# C vs Number of Grid Points
# ================================================================ #
f, ax1 = plt.subplots(1, 1, figsize=(11, 7), facecolor='w')
rc('font', **{'size': 15})

f.suptitle(f"C vs Number of Grid Points", fontsize=30)
ax1.tick_params(axis='both', which='major', labelsize=20)

C = np.array([0.0068, 0.0126, 0.0423, 0.2215])
N = np.array([20, 40, 80, 160])**2

alpha, beta = curve_fit(lambda t,a,b: b*(t**a), N, C, p0=(1,0.000001))[0]
x = np.linspace(0, N[-1], 100)

ax1.plot(x, beta*(x**alpha), color="Red", linewidth=2, linestyle="--")
ax1.scatter(N, C, color="Black", s=90)
ax1.grid()
ax1.set_xlabel("# pf Points")
ax1.set_ylabel("C (seconds/step)")
ax1.semilogy()
ax1.semilogx()

# ================================================================ #
# U Line Sampling at Re=100, 200, 400, 500
# ================================================================ #

N = 40 # Change to match grid size
sim="re100" # Change to match which simulation to parse

top_x_u = np.zeros((N, 2))
mid_x_u = np.zeros((N, 2))
btm_x_u = np.zeros((N, 2))

with open(f"../../sims/{sim}_cavity/postProcessing/sample/0.5/top_line_U.xy") as f:
    lines = f.readlines()
    for index, line in enumerate(lines):
        splits = line.split(" ")
        top_x_u[index] = [float(splits[0]), float(splits[3])]
        
with open(f"../../sims/{sim}_cavity/postProcessing/sample/0.5/mid_line_U.xy") as f:
    lines = f.readlines()
    for index, line in enumerate(lines):
        splits = line.split(" ")
        mid_x_u[index] = [float(splits[0]), float(splits[3])]
        
with open(f"../../sims/{sim}_cavity/postProcessing/sample/0.5/bottom_line_U.xy") as f:
    lines = f.readlines()
    for index, line in enumerate(lines):
        splits = line.split(" ")
        btm_x_u[index] = [float(splits[0]), float(splits[3])]


f, ax1 = plt.subplots(1, 1, figsize=(11, 7), facecolor='w')
rc('font', **{'size': 15})

f.suptitle(f"U Line Sampling at Re=400", fontsize=30)
ax1.tick_params(axis='both', which='major', labelsize=20)

weights = np.zeros(top_x_u[:, 1].shape) + 0.001
weights[0] = 1
x = np.linspace(0, 0.1, 50)

mm, m, b = curve_fit(lambda t,mm,m,b: mm*(t**2) + m*t + b, top_x_u[:, 0], top_x_u[:, 1], sigma=weights)[0]
ax1.plot(x, mm*(x**2) + m*x + b, color="Red")
print(f"u(x, 0.099) = {round(mm, 2)}x^2 + {round(m, 2)}x + {round(b,2)}")

mm, m, b = curve_fit(lambda t,mm,m,b: mm*(t**2) + m*t + b, mid_x_u[:, 0], mid_x_u[:, 1], sigma=weights)[0]
ax1.plot(x, mm*(x**2) + m*x + b, color="Green")
print(f"u(x, 0.099) = {round(mm, 2)}x^2 + {round(m, 2)}x + {round(b,2)}")

mm, m, b = curve_fit(lambda t,mm,m,b: mm*(t**2) + m*t + b, btm_x_u[:, 0], btm_x_u[:, 1], sigma=weights)[0]
ax1.plot(x, mm*(x**2) + m*x + b, color="Blue")
print(f"u(x, 0.099) = {round(mm, 2)}x^2 + {round(m, 2)}x + {round(b,2)}")

ax1.scatter(top_x_u[:, 0], top_x_u[:, 1], color="Red", label="y=0.099", s=90)
ax1.scatter(mid_x_u[:, 0], mid_x_u[:, 1], color="Green", label="y=0.09", s=90)
ax1.scatter(btm_x_u[:, 0], btm_x_u[:, 1], color="Blue", label="y=0.08", s=90)
ax1.grid()
ax1.set_xlabel("x")
ax1.set_ylabel("u")
ax1.set_xlim(0, 0.1)
ax1.legend(loc="upper right")

# ================================================================ #
# u Component Along Various Horizontal Lines
# ================================================================ #

f, ax1 = plt.subplots(1, 1, figsize=(11, 7), facecolor='w')
rc('font', **{'size': 15})

f.suptitle(f"u Component Along Various Horizontal Lines", fontsize=30)
ax1.tick_params(axis='both', which='major', labelsize=20)

ax1.scatter([0.099]*N, top_x_u[:, 1], color="Red", label="y=0.099", s=90)
ax1.scatter([0.09]*N, mid_x_u[:, 1], color="Green", label="y=0.09", s=90)
ax1.scatter([0.08]*N, btm_x_u[:, 1], color="Blue", label="y=0.08", s=90)

u_vals = np.zeros(3*N)
u_vals[0:N] = top_x_u[:, 1]
u_vals[N:2*N] = mid_x_u[:, 1]
u_vals[2*N:3*N] = btm_x_u[:, 1]

n = 10000
y_vals = np.array([0.1]*n + [0.099]*N + [0.09]*N + [0.08]*N)
du_vals = np.zeros(u_vals.size + n) + 1
du_vals[n:] = u_vals

mm, m, b = curve_fit(lambda t,mm,m,b: mm*(t**2) + m*t + b, y_vals, du_vals)[0]

ys = np.linspace(0.07, 1, 1000)
ax1.plot(ys, mm*(ys**2) + m*ys+b, color="Black", linewidth=3, linestyle="-")

ax1.grid()
ax1.set_ylabel("u")
ax1.set_xlabel("y")
ax1.set_xlim(0.075, 0.1)
ax1.set_ylim(-0.2, 1)
ax1.legend()

# ================================================================ #
# Shear Stress vs. Reynolds Number
# ================================================================ #

N = 20
sim="o1"

def getSim(sim, N):
    top_x_u = np.zeros((N, 2))
    mid_x_u = np.zeros((N, 2))
    btm_x_u = np.zeros((N, 2))
    with open(f"../../sims/{sim}_cavity/postProcessing/sample/0.5/top_line_U.xy") as f:
        lines = f.readlines()
        for index, line in enumerate(lines):
            splits = line.split(" ")
            top_x_u[index] = [float(splits[0]), float(splits[3])]

    with open(f"../../sims/{sim}_cavity/postProcessing/sample/0.5/mid_line_U.xy") as f:
        lines = f.readlines()
        for index, line in enumerate(lines):
            splits = line.split(" ")
            mid_x_u[index] = [float(splits[0]), float(splits[3])]

    with open(f"../../sims/{sim}_cavity/postProcessing/sample/0.5/bottom_line_U.xy") as f:
        lines = f.readlines()
        for index, line in enumerate(lines):
            splits = line.split(" ")
            btm_x_u[index] = [float(splits[0]), float(splits[3])]
    return (top_x_u, mid_x_u, btm_x_u)

dudy = []
iuy = []
for N, sim in [(40, "re100"), (40, "re200"), (40, "re400"), (40, "re500")]:    
    top_x_u, mid_x_u, btm_x_u = getSim(sim, N)
    u_vals = np.zeros(3*N)
    u_vals[0:N] = top_x_u[:, 1]
    u_vals[N:2*N] = mid_x_u[:, 1]
    u_vals[2*N:3*N] = btm_x_u[:, 1]

    n = 10000
    y_vals = np.array([0.1]*n + [0.099]*N + [0.09]*N + [0.08]*N)
    du_vals = np.zeros(u_vals.size + n) + 1
    du_vals[n:] = u_vals

    mm, m, b = curve_fit(lambda t,mm,m,b: mm*(t**2) + m*t + b, y_vals, du_vals)[0]
    dudy.append(2*mm*1 + m)
    iuy.append((mm*(1**2) + m*1) - (mm*(0**2) + m*0))
    
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 7), facecolor='w')
rc('font', **{'size': 15})

ax1.set_title(f"Shear Stress vs. Reynolds Number", fontsize=30)
ax1.tick_params(axis='both', which='major', labelsize=20)

ax1.scatter([100, 200, 400, 500], dudy, color="Black", s=90)
m, b = np.polyfit([100, 200, 400, 500], dudy, deg=1)
x = np.linspace(0, 500, 10)

ax1.plot(x, m*x+b, color="Red", linewidth=2)
ax1.grid()
ax1.set_xlim(0, 500)
ax1.set_ylim(2000, 4500)
ax1.set_xlabel("Re")
ax1.set_ylabel("Nondimensional Stress")

ax2.set_title(f"Force vs. Reynolds Number", fontsize=30)
ax2.tick_params(axis='both', which='major', labelsize=20)

ax2.scatter([100, 200, 400, 500], iuy, color="Black", s=90)
m, b = np.polyfit([100, 200, 400, 500], iuy, deg=1)
x = np.linspace(0, 500, 10)

ax2.plot(x, m*x+b, color="Red", linewidth=2)
ax2.grid()
ax2.set_xlim(0, 500)
ax2.set_xlabel("Re")
ax2.set_ylabel("Nondimensional Force")
