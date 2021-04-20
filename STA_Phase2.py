"""
Created on Sat Apr 17 01:43:44 2021

@author: Lorenza Mottinelli
"""

import numpy as np
from scipy import optimize
from scipy import integrate
from matplotlib import pyplot as plt
import matplotlib
from matplotlib import colors as col
from matplotlib import colorbar as cbar
import matplotlib.cm as cm
from math import *
matplotlib.use('Qt5Agg')

# constants for Earth Sun system
G = 6.67408*10**-11 # SI units
m_sun = 1.9891e+30  # kg
m_earth = 6.0477e+24  # kg
AU = 149597870.7e03  # m
sigma = 1.53  # g/m2
mu_ES = m_earth / (m_earth + m_sun)
x_L2_Deccia = 151105099.2  # km
mu_sun = 132717800000e09  # m3/s2 Suns grav parameter #mgod
omega = [0, 0, 1]
t_star = np.sqrt(AU**3 / (G*(m_earth + m_sun)))
asteroid = np.loadtxt('2015_YQ1_frameS_2025_2030.dat')
# compute L2 (task 1.1a)
def l2(x):
    return x - ((1 - mu_ES) / (x + mu_ES) ** 3) * (x + mu_ES) - (mu_ES / (x - (1 - mu_ES)) ** 3) * (x - (1 - mu_ES))


x_L2_dl = optimize.newton(l2, 1, tol=1e-12)  # dimensionless
x_L2 = x_L2_dl * AU #m
r_L2 = [x_L2_dl, 0, 0]
r1_L2 = [x_L2_dl + mu_ES, 0, 0]
r2_L2 = [x_L2_dl - (1 - mu_ES), 0, 0]
print('x_L2 diensionless:', x_L2_dl)
print('x_L2:', x_L2, "[km]")
print('Difference vs Deccia:', abs(x_L2 - x_L2_Deccia), '[km]')


# compute solar sail performance beta (task 1.2a), compute solar sail required orientation (task 1.2b)
def get_artificial_Lagrange(r):
    r1 = [r[0] + mu_ES, r[1], r[2]]
    r2 = [r[0] - (1 - mu_ES), r[1], r[2]]
    divU = np.divide(np.multiply(r1, (1 - mu_ES)), np.linalg.norm(r1) ** 3) + np.divide(np.multiply(r2, mu_ES),
                                                                                        np.linalg.norm(
                                                                                            r2) ** 3) + np.cross(omega,
                                                                                                                 np.cross(
                                                                                                                     omega,
                                                                                                                     r))
    print('divU:', divU)
    n_hat = divU / np.linalg.norm(divU)
    beta = np.linalg.norm(r1) ** 2 / (1 - mu_ES) * np.dot(divU, n_hat) / np.dot(r1 / np.linalg.norm(r1), n_hat) ** 2
    return n_hat, beta


deltaX = 0.011
r = [x_L2_dl - deltaX, 0, 0]
n_hat, beta = get_artificial_Lagrange(r)
print('beta:', beta, '[-]')
print('n_hat:', n_hat)

# verify the correctness of the lightness number and sail orientation (task 1.2c)
r_veri = [0.95, 0, 0.1]
n_hat_veri, beta_veri = get_artificial_Lagrange(r_veri)
print('beta_veri:', beta_veri, '[-]')
print('n_hat_veri:', n_hat_veri)

# compute the required solar-sail area (task 1.2d)
m_sat = 10  # kg
beta = beta
r_scalar_meters = np.linalg.norm(r) * AU
solar_int_earth = 1367  # W/m2 https://www.sciencedirect.com/topics/engineering/solar-radiation-pressure
lightspeed = 2.99792458e08  # m/s https://www.google.com/search?q=speed+of+light&rlz=1C1CHBF_enNL927NL927&oq=speed+of+light&aqs=chrome..69i57j6j0j46j0l6.12924j1j4&sourceid=chrome&ie=UTF-8
p = solar_int_earth / (
        (r[0] ** 2) * lightspeed)  # https://www.sciencedirect.com/topics/engineering/solar-radiation-pressure
side_length = np.sqrt((m_sat * beta * mu_sun) / (2 * p * r_scalar_meters ** 2))
print('side length:', side_length, '[m]')

# assess the stability of your Lagrange point (task 2.1a,b)
r1_L2_norm = np.linalg.norm(r1_L2)
r2_L2_norm = np.linalg.norm(r2_L2)
uxx = -1 + (1 - mu_ES) * ((r1_L2_norm ** 3 - (x_L2_dl + mu_ES) * 3 * r1_L2_norm ** 2) / r1_L2_norm ** 6) + mu_ES * (
        (r2_L2_norm ** 3 - (x_L2_dl - (1 - mu_ES)) * 3 * r2_L2_norm ** 2) / r2_L2_norm ** 6)
uxy = 0
uyy = -1 + (1 - mu_ES) * ((r1_L2_norm ** 3) / r1_L2_norm ** 6) + mu_ES * ((r2_L2_norm ** 3) / r2_L2_norm ** 6)
A = [[0, 0, 1, 0], [0, 0, 0, 1], [-uxx, -uxy, 0, 2], [-uxy, -uyy, -2, 0]]
eigvals, eigvecs = np.linalg.eig(A)
print('eigenvalues:',eigvals)  # https://descanso.jpl.nasa.gov/monograph/series12/LunarTraj--Overall.pdf pg 110 for veri
print('eigenvectors:', eigvecs)

# compute and plot the unstable and stable manifolds of your Lagrange point (task 2.2)

def deriv(Xi,t,mu_ES):
    r1_L2_pert = [Xi[0] + mu_ES, Xi[1]]
    r1_L2_pert_norm = np.linalg.norm(r1_L2_pert)
    r2_L2_pert = [Xi[0] - (1 - mu_ES), Xi[1]]
    r2_L2_pert_norm = np.linalg.norm(r2_L2_pert)
    dudx = Xi[0] - ((1 - mu_ES)/r1_L2_pert_norm**3)*(Xi[0] + mu_ES) - (mu_ES/r2_L2_pert_norm**3)*(Xi[0]- (1-mu_ES))
    acc_x = dudx + 2*Xi[3]
    dudy =  + Xi[1] - Xi[1]*((1-mu_ES)/r1_L2_pert_norm**3) - Xi[1]*mu_ES/r2_L2_pert_norm**3
    acc_y = dudy - 2*Xi[2]
    dxdt = [Xi[2],Xi[3],acc_x,acc_y]
    return dxdt

# define initial and perturbed state
X0 = [r_L2[0],r_L2[1], 0, 0]
epsilon = 10 ** -5
Xi_fw_pos = X0 + np.multiply(epsilon, np.real(eigvecs[:, 1]))
Xi_fw_neg = X0 - np.multiply(epsilon, np.real(eigvecs[:, 1]))
Xi_bw_pos = X0 + np.multiply(epsilon, np.real(eigvecs[:, 0]))
Xi_bw_neg = X0 - np.multiply(epsilon, np.real(eigvecs[:, 0]))
t_start = 0/t_star
t_end = (3*365.25*24*3600)/t_star
t_fw = np.linspace(t_start,t_end,10000)
t_bw = np.linspace(t_end,t_start,1000)
fw_pos = integrate.odeint(deriv, Xi_fw_pos, t_fw, args = (mu_ES,),atol=1e-12,rtol=1e-12)
fw_neg = integrate.odeint(deriv, Xi_fw_neg, t_fw, args = (mu_ES,),atol=1e-12,rtol=1e-12)
bw_pos = integrate.odeint(deriv, Xi_bw_pos, t_bw, args = (mu_ES,),atol=1e-12,rtol=1e-12)
bw_neg = integrate.odeint(deriv, Xi_bw_neg, t_bw, args = (mu_ES,),atol=1e-12,rtol=1e-12)

#%% plot manifolds L2
plt.figure()
plt.plot(fw_pos[:,0],fw_pos[:,1],label = 'unstable invariant manifold,\n positive perturbation')
plt.plot(fw_neg[:,0],fw_neg[:,1],label = 'unstable invariant manifold,\n negative perturbation')
plt.plot(bw_pos[:,0],bw_pos[:,1],label = 'stable invariant manifold, \n positive perturbation')
plt.plot(bw_neg[:,0],bw_neg[:,1],label = 'stable invariant manifold,\n negative perturbation')
plt.plot(x_L2_dl,0,'xk',color = 'purple',label = 'L2')
plt.ylabel('y [AU]')
plt.xlabel('x [AU]')
plt.xlim([-1.5, 1.5])
plt.ylim([-1.5, 1.5])
plt.title('Invariant manifolds of Earth- Sun L2 point')
plt.legend()
plt.grid()
plt.show()


#%% Plot the trajectory of your assigned co-orbital asteroid (task 3.1a)
plt.figure()
plt.plot(fw_neg[:,0],fw_neg[:,1],label = 'unstable invariant manifold,\n negative perturbation')
plt.plot(bw_pos[:,0],bw_pos[:,1],label = 'stable invariant manifold, \n positive perturbation')
plt.plot(asteroid[:,1],asteroid[:,2], label = '2015 YQ1 trajectory')
plt.plot(-mu_ES,0,'o',color = 'gold',label = 'Sun',markersize = 12)
plt.plot(1-mu_ES,0,'o',color = 'blue',label = 'Earth',markersize = 8)
plt.plot(x_L2_dl,0,'xk',color = 'purple',label = 'L2')
plt.ylabel('y [AU]')
plt.xlabel('x [AU]')
plt.xlim([-1.5, 1.5])
plt.ylim([-1.5, 1.5])
plt.title('2015 YQ1 asteroid trajectory and manifolds')
plt.legend()
plt.grid()
plt.show()


#%% Propagate (one of) the unstable manifolds with ideal solar-sail acceleration (task 3.2a)
def deriv_solar_sail(Xi,t_fw,mu_ES,alpha,delta,beta):
    r1_L2_pert = [(Xi[0] + mu_ES), Xi[1]]
    r1_L2_pert_norm = np.linalg.norm(r1_L2_pert)
    r2_L2_pert = [(Xi[0] - (1 - mu_ES)), Xi[1]]
    r2_L2_pert_norm = np.linalg.norm(r2_L2_pert)

    dudx = Xi[0] - ((1 - mu_ES)/r1_L2_pert_norm**3)*(Xi[0] + mu_ES) - mu_ES * ((Xi[0] - (1-mu_ES))/r2_L2_pert_norm**3)
    acc_x = 2*Xi[3] + dudx
    dudy =  Xi[1] - Xi[1]*(1-mu_ES)/(r1_L2_pert_norm**3) - (Xi[1]*mu_ES)/r2_L2_pert_norm**3
    acc_y = - 2*Xi[2] + dudy

    acc_solar_x = beta* (1-mu_ES) / r1_L2_pert_norm**2 * (cos(alpha))**2 * (
                ((Xi[0] + mu_ES) / r1_L2_pert_norm) * np.cos(cone_angle) - (
                    Xi[1] / r1_L2_pert_norm * np.sin(cone_angle) * np.sin(delta)))

    acc_solar_y = beta * (1 - mu_ES) / (r1_L2_pert_norm ** 2) * (cos(alpha))**2 *(
                (Xi[1] / r1_L2_pert_norm) * np.cos(cone_angle) + (
                    (Xi[0] + mu_ES) / r1_L2_pert_norm * np.sin(cone_angle) * np.sin(delta)))

    dxdt = [Xi[2], Xi[3], (acc_x+acc_solar_x), (acc_y+acc_solar_y)]

    return dxdt

epsilon = 10 ** -5
Xi_fw_neg = X0 - np.multiply(epsilon, np.real(eigvecs[:, 1])) # unstable neg manifold leading to asteroid
delta = np.deg2rad(90)
alpha_step = 2.5
cone_angles = np.arange(np.deg2rad(-90), np.deg2rad(90+alpha_step), np.deg2rad(alpha_step))
t_start = 0/t_star
t_end = (3*365.25*24*3600)/t_star
t_fw = np.linspace(t_start, t_end, 1000)
beta = 0.01
plt.figure()
cmap = plt.get_cmap('jet', len(cone_angles))
traj_ss = []
for i in range(len(cone_angles)):
    cone_angle = cone_angles[i]
    print('alpha:', np.rad2deg(cone_angle))
    traj_ss.append(integrate.odeint(deriv_solar_sail, Xi_fw_neg, t_fw, args=(mu_ES, cone_angle, delta, beta), atol=1e-12, rtol=1e-12,Dfun=None))
    plt.plot(traj_ss[-1][:, 0], traj_ss[-1][:, 1], c=cmap(i))
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=np.rad2deg(min(cone_angles)), vmax=np.rad2deg(max(cone_angles))))
plt.colorbar(sm,label = r'cone angle $\alpha$ [degrees]')
plt.plot(fw_neg[:,0],fw_neg[:,1],label = 'unstable invariant manifold,\n negative perturbation')
plt.plot(bw_pos[:,0],bw_pos[:,1],label = 'stable invariant manifold, \n positive perturbation')
plt.plot(asteroid[:,1],asteroid[:,2], label = '2015 YQ1 trajectory')
plt.plot(-mu_ES,0,'o',color = 'gold',label = 'Sun',markersize = 12)
plt.plot(1-mu_ES,0,'o',color = 'blue',label = 'Earth',markersize = 8)
plt.plot(x_L2_dl,0,'xk',color = 'purple',label = 'L2')
plt.ylabel('y [AU]')
plt.xlabel('x [AU]')
plt.xlim([-1.5, 1.5])
plt.ylim([-1.5, 1.5])
plt.title('Solar sail propelled invariant manifolds')
plt.legend()
plt.grid()
plt.show()

#%% Quantify the performance of solar-sail propulsion, flyby distance (task 3.2b)

def get_flyby_info(traj_sat, traj_asteroid):
    dist_all = []
    vel_all = []
    x_unprop,y_unprop,vx_unprop,vy_unprop = traj_sat[:, 0],traj_sat[:, 1],traj_sat[:, 2],traj_sat[:, 3]
    for point in traj_asteroid:
        x_ast,y_ast,vx_ast,vy_ast = point[1], point[2], point[3], point[4]
        dist_pt = np.multiply(np.linalg.norm([(x_unprop - x_ast), (y_unprop - y_ast)], axis=0),AU/1000)#km
        vel_pt = np.multiply(np.linalg.norm([(vx_unprop - vx_ast), (vy_unprop - vy_ast)], axis=0),AU/t_star) #m/s
        for k in range(len(dist_pt)):
            if dist_pt[k] <= 150000: #km
                dist_all.append(dist_pt[k])
                vel_all.append(vel_pt[k])
    index_min = np.argmin(vel_all)
    distance_min = dist_all[index_min]
    velocity_min = vel_all[index_min]
    return distance_min, velocity_min

# unpropelled
flyby_fw_neg = get_flyby_info(fw_neg, asteroid)
# flyby_bw_pos = flyby_unprop(bw_pos, asteroid)
print('Unstable manifold: min flyby distance = ',flyby_fw_neg[0],'km','min flyby vel =',flyby_fw_neg[1],'m/s')
# print('Stable manifold: min flyby distance = ',flyby_bw_pos[0],'min flyby vel =',flyby_bw_pos[1])

# solar sail
mindist_alphas = []
minvel_alphas = []
for i in range(len(cone_angles)):
    distance_min_i, velocity_min_i = get_flyby_info(traj_ss[i], asteroid)
    minvel_alphas.append(velocity_min_i)
best_alpha_min_vel  = cone_angles[minvel_alphas,index(min(minvel_alphas))]
print(best_alpha_for_vel)