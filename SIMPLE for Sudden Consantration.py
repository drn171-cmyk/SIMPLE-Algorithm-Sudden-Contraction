# This code is for solving laminar flow with  sudden consentration by applying SIMPLE

import numpy as np
import matplotlib.pyplot as plt
import numpy.ma as ma


def tdma(a, b, c, d):
    n = len(d)
    c_p = np.zeros(n)
    d_p = np.zeros(n)
    x = np.zeros(n)
    
    # Forward Sweep
    c_p[0] = c[0] / b[0]
    d_p[0] = d[0] / b[0]
    
    for i in range(1, n):
        m = 1.0 / (b[i] - a[i] * c_p[i-1])
        c_p[i] = c[i] * m
        d_p[i] = (d[i] - a[i] * d_p[i-1]) * m
        
    # Back Substitution
    x[-1] = d_p[-1]
    for i in range(n-2, -1, -1):
        x[i] = d_p[i] - c_p[i] * x[i+1]
        
    return x


d=0.005# m
D=0.01# m 
L = 10*D # m
l = 10*d # m

nu = 1.5e-5 # m^2/s
U =0.1 # m/s
P = 0 # Pa

print(f' Reynolds number : {U*D/nu}')

N=50 # number of grid points for y direction
h=D/N/2 # grid spacing
M1 = int(20*N)#number of grid points for x direction through the large part 
M2 = int(20*d/D*N) #number of grid points for x direction through the small part 

tol=1e-3 # tolerance for convergence 
min_iter = 200 # don't check convergence before this many iterations (avoids false-early exit)
max_iter = 3000 # maximum number of iterations
iter = 0

# Defining error matrixes
err_u=np.zeros(max_iter+2)
err_v=np.zeros(max_iter+2)
err_p=np.zeros(max_iter+2)

err=np.zeros(max_iter+2)
err[0]=1

omega1 = .8 # Relaxation factor for velocity
omega2 = .3 # Relaxation factor for pressure

# Defining velocity and pressure matrixes
u = np.zeros((N+2, M1+M2+2))
v = np.zeros((N+2, M1+M2+2))
p = np.zeros((N+2, M1+M2+2))


contact_height=int(N*d/D)

# Fill large channel with inlet velocity
u[:, :M1+1] = U 

# Fill small channel with velocity appropriate to mass conservation (A1*V1 = A2*V2)
u_narrow = U * (D / d)
u[:contact_height+1, M1+1:] = u_narrow

u_old = u.copy()
v_old = v.copy()
p_old = p.copy()

# Boundary Conditions for the inlet and outlet
u[:,0]=U
v[:,0]=0
p[:,-1]=P

alpha=h**2/(4*nu)
beta=nu/h**2

k=1

total_cells = N*(M1+M2) - (N-contact_height)*M2

while (err[k-1] > tol or iter < min_iter) and iter<max_iter:
    
  
    err_u[k]=0
    err_v[k]=0

  
    for i in range(1,N+1):
        for j in range(1,M1+M2+1):
            
            # Wall boundary condition
            if j > M1 and i > contact_height:
                u[i,j]=0
                v[i,j]=0
                continue

            # Hybrid differencing for stability (prevents NaN at high U)

            a1=max(beta-u_old[i,j]/(2*h), 0)
            a2=max(beta+u_old[i,j]/(2*h), 0)
            a3=max(beta-v_old[i,j]/(2*h), 0)
            a4=max(beta+v_old[i,j]/(2*h), 0)
            diag = a1+a2+a3+a4

            u[i,j]=(a1*u_old[i,j+1]+a2*u[i,j-1]+
                        a3*u_old[i+1,j]+a4*u[i-1,j]-
                        (p_old[i,j+1]-p_old[i,j-1])/(2*h))/diag

            u[i,j]=omega1*u[i,j]+(1-omega1)*u_old[i,j]

            err_u[k]=(u[i,j]-u_old[i,j])**2+err_u[k]

            v[i,j]=(a1*v_old[i,j+1]+a2*v[i,j-1]+
                        a3*v_old[i+1,j]+a4*v[i-1,j]-
                        (p_old[i+1,j]-p_old[i-1,j])/(2*h))/diag

            v[i,j]=omega1*v[i,j]+(1-omega1)*v_old[i,j]
   
            err_v[k]=(v[i,j]-v_old[i,j])**2+err_v[k]

    # Boundary Conditions

    u[0,:]=u[2,:] # Symmetry Boundary Condition
    v[0,:]=0 # Symmetry Boundary Condition
    v[1,:]=0 # Symmetry Boundary Condition
    u[:,-1]=u[:,-2] # Outlet zero gradient
    v[:,-1]=v[:,-2] # Outlet zero gradient
    
    # Inlet
    u[:,0]=U
    v[:,0]=0

    # Error Calculation
    err_u[k]=1/total_cells*err_u[k]
    err_u[k]=np.sqrt(err_u[k])
    err_v[k]=1/total_cells*err_v[k]
    err_v[k]=np.sqrt(err_v[k])

    b = np.zeros((N+2, M1+M2+2))
    p_prime = np.zeros((N+2, M1+M2+2))

    for i in range(1,N+1):
        for j in range(1,M1+M2+1):
            if j > M1 and i > contact_height:
                continue
            b[i,j]=(u[i,j+1]-u[i,j-1])/(2*h) + (v[i+1,j]-v[i-1,j])/(2*h)

    # Poisson Iteration (TDMA / Line-by-Line ADI Method)
    for _ in range(5): # 50 yerine sadece 5 tarama!
        
        # 1. Yatay Tarama (X-Sweep)
        for i in range(1, N+1):
            n_cells = M1 + M2
            a_td = -np.ones(n_cells)
            b_td = 4 * np.ones(n_cells)
            c_td = -np.ones(n_cells)
            d_td = np.zeros(n_cells)
            
            for j in range(1, n_cells + 1):
                idx = j - 1
                if j > M1 and i > contact_height:
                    a_td[idx], b_td[idx], c_td[idx] = 0, 1, 0
                    d_td[idx] = 0
                else:
                    # RHS (Sağ taraf): Kuzey, Güney ve Kaynak terimi
                    d_td[idx] = p_prime[i+1, j] + p_prime[i-1, j] - (h**2 / alpha) * b[i, j]
            
            # Boundary Conditions (X axis)
            b_td[0] = 3 # Zero gradient at inlet (p_0 = p_1)
            c_td[-1] = 0 # Outlet pressure is fixed to 0, no east neighbor
            
            # Tridiagonal solve 
            p_prime[i, 1:n_cells+1] = tdma(a_td, b_td, c_td, d_td)
            
        # 2. Vertical Sweep (Y-Sweep)
        for j in range(1, M1 + M2 + 1):
            n_cells = N
            a_td = -np.ones(n_cells)
            b_td = 4 * np.ones(n_cells)
            c_td = -np.ones(n_cells)
            d_td = np.zeros(n_cells)
            
            for i in range(1, n_cells + 1):
                idx = i - 1
                if j > M1 and i > contact_height:
                    a_td[idx], b_td[idx], c_td[idx] = 0, 1, 0
                    d_td[idx] = 0
                else:
                    # RHS: East, West and Source term
                    d_td[idx] = p_prime[i, j+1] + p_prime[i, j-1] - (h**2 / alpha) * b[i, j]
            
            # Boundary Conditions (Y axis)
            b_td[0] = 3 # Symmetry axis (p_0 = p_1)
            b_td[-1] = 3 # Top wall (p_N+1 = p_N)
            
            # Tridiagonal solve 
            p_prime[1:n_cells+1, j] = tdma(a_td, b_td, c_td, d_td)
        
        # Boundary Conditions
        p_prime[:,0]=p_prime[:,1]
        p_prime[:,-1]= 0.0  
        p_prime[0,:]=p_prime[1,:]
        p_prime[-1,:]=p_prime[-2,:]

    #Update velocity
    for i in range(1,N+1):
        for j in range(1,M1+M2+1):
            if j > M1 and i > contact_height:
                continue
            
            u[i,j] = u[i,j] - alpha * (p_prime[i,j+1] - p_prime[i,j-1]) / (2*h)
            

            v[i,j] = v[i,j] - alpha * (p_prime[i+1,j] - p_prime[i-1,j]) / (2*h)

    # Boundary Conditions
    u[0,:]=u[2,:] 
    v[0,:]=0 
    v[1,:]=0 
    u[:,-1]=u[:,-2] 
    v[:,-1]=v[:,-2] 
    u[:,0]=U
    v[:,0]=0

    #Update pressure
    p_old = p.copy()
    p = p + omega2 * p_prime
    
    err[k] = max(err_u[k], err_v[k])
    
    u_old = u.copy()
    v_old = v.copy()
    
    # Print Error Every 50 iterations
    if k % 10 == 0:
        print(f"Iter: {k}, Error: {err[k]:.6f}")
    
    iter += 1
    k += 1
            
# Post-processing and visualization

x=np.arange(0,M1+M2+2)*h
y=np.arange(0,N+2)*h
X,Y=np.meshgrid(x,y)

V_mag=np.sqrt(u**2+v**2)

# Masking the solid wall region
wall_mask=(Y>contact_height*h)&(X>M1*h)

V_mag_masked=ma.masked_where(wall_mask,V_mag)
u_masked=ma.masked_where(wall_mask,u)
v_masked=ma.masked_where(wall_mask,v)

# Convergence history plot
plt.figure(figsize=(7,4))
plt.semilogy(range(1,k), err[1:k])
plt.axhline(tol, color='r', linestyle='--', label=f'tol={tol:.0e}')
plt.xlabel('Iteration')
plt.ylabel('Error (log scale)')
plt.title('Convergence History')
plt.legend()
plt.show()

# Velocity contour plot
plt.figure(figsize=(12,4))
plt.contourf(X,Y,V_mag_masked,levels=100,cmap='jet')
plt.colorbar(label='Velocity Magnitude (m/s)')

# Drawing wall boundaries and symmetry line
plt.plot([M1*h,M1*h,x.max()],[y.max(),contact_height*h,contact_height*h],'k-',linewidth=3)
plt.plot([0,x.max()],[0,0],'k--',linewidth=1.5,label='Symmetry')

plt.title('Velocity Contour')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.show()

# Streamlines plot
plt.figure(figsize=(12,4))
plt.streamplot(X,Y,u_masked,v_masked,color=V_mag_masked,cmap='jet',density=1.5)
plt.colorbar(label='Velocity Magnitude (m/s)')

# Shading the solid wall
plt.fill_between([M1*h,x.max()],contact_height*h,y.max(),color='gray',alpha=0.5)
plt.plot([0,x.max()],[0,0],'k--',linewidth=1.5,label='Symmetry')

plt.title('Streamlines')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.show()



            


            


            
            




    

