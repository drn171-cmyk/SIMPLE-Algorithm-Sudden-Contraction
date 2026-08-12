# This code is for solving laminar flow with  sudden consentration by applying SIMPLE

import numpy as np
import matplotlib.pyplot as plt
import numpy.ma as ma

d=0.005# m
D=0.01# m 
L = 10*D # m
l = 10*d # m

nu = 1.5e-5 # m^2/s
U =2 # m/s
P = 0 # Pa

print(f' Reynolds number : {U*D/nu}')

N=50 # number of grid points for y direction
h=D/N/2 # grid spacing
M1 = int(20*N)#number of grid points for x direction through the large part 
M2 = int(20*d/D*N) #number of grid points for x direction through the small part 

tol=1e-4 # tolerance for convergence 
max_iter = 1000 # maximum number of iterations
iter = 0

# Defining error matrixes
err_u=np.zeros(max_iter)
err_v=np.zeros(max_iter)
err_p=np.zeros(max_iter)

err=np.zeros(max_iter)
err[0]=1

omega1 = .5 # Relaxation factor for velocity
omega2 = .3 #Relaxation factor for pressure

# Defining velocity and pressure matrixes
u = np.zeros((N+2, M1+M2+2))
v = np.zeros((N+2, M1+M2+2))
p = np.zeros((N+2, M1+M2+2))

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
contact_height=int(N*d/D)

while err[k-1] > tol and iter<max_iter:
    
  
    err_u[k]=0
    err_v[k]=0

  
    for i in range(1,N+1):
        for j in range(1,M1+M2+1):
            
            # Wall boundary condition
            if j > M1 and i > contact_height:
                u[i,j]=0
                v[i,j]=0
                continue
            
         
            a1=beta-u_old[i,j]/(2*h)
            a2=beta+u_old[i,j]/(2*h)
            a3=beta-v_old[i,j]/(2*h)
            a4=beta+v_old[i,j]/(2*h)

    
            u[i,j]=alpha*(a1*u_old[i,j+1]+a2*u[i,j-1]+
                        a3*u_old[i+1,j]+a4*u[i-1,j]-
                        (p_old[i,j+1]-p_old[i,j-1])/(2*h))

            u[i,j]=omega1*u[i,j]+(1-omega1)*u_old[i,j]

            err_u[k]=(u[i,j]-u_old[i,j])**2+err_u[k]

            v[i,j]=alpha*(a1*v_old[i,j+1]+a2*v[i,j-1]+
                        a3*v_old[i+1,j]+a4*v[i-1,j]-
                        (p_old[i+1,j]-p_old[i-1,j])/(2*h))

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
    toplam_hucre = (contact_height)*M1 + (contact_height)*M2
    err_u[k]=1/toplam_hucre*err_u[k]
    err_u[k]=np.sqrt(err_u[k])
    err_v[k]=1/toplam_hucre*err_v[k]
    err_v[k]=np.sqrt(err_v[k])

    b = np.zeros((N+2, M1+M2+2))
    p_prime = np.zeros((N+2, M1+M2+2))

    for i in range(1,N+1):
        for j in range(1,M1+M2+1):
            if j > M1 and i > contact_height:
                continue
            b[i,j]=(u[i,j+1]-u[i,j-1])/(2*h) + (v[i+1,j]-v[i-1,j])/(2*h)

    # Poisson Iteration
    for _ in range(50):
        p_prime_old = p_prime.copy()
        for i in range(1,N+1):
            for j in range(1,M1+M2+1):
                if j > M1 and i > contact_height:
                    continue

                p_prime[i,j]=0.25*(p_prime_old[i,j+1]+p_prime[i,j-1]+
                                 p_prime_old[i+1,j]+p_prime[i-1,j]-
                                 (h**2/alpha)*b[i,j])
        
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
            
            



            


            


            
            




    

