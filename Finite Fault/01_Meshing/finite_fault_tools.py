import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from matplotlib import path
from termcolor import colored

#    mesh_fault_plane
#    This function mesh the fault plane. The convention of the fucntion is due to Aki and Richard, 2nd edition,  Figure 4.20,  page 109
#    Thus X is in NS (increased northerly), y in EW (increases easterly) and z positive in downward direction
#    The Coordinates are given for top-center of each subfault or segment: |'|
#    Origin of local coordinate system at hypocenter: X (NS) = 0,  Y (EW) = 0
#
#    For meshing the fault plane,  we consider two basis vectors, one in strike direction and other in dip direction
#    v_strike=(cos(fi), sin(fi), 0)
#    v_dip=(-cos(delta)sin(fi), cos(delta)cos(fi), sin(delta))   positive in downward direction
#    
#    Input:  strike:     strike angle in degree  =fi
#            dip:                        dip angle in degree  = delta
#            n_strike:                number of element in strike direction
#            n_dip:                    number of element in dip direction
#            len_elem_strike:    length of elements in strike direction in km
#            len_elem_dip:       length of elements in dip direction in km
#            htop:                     length to top of the fault plane in km
#            ls_hyp:                  distance of hypocenter to upper left corner of the fault plane in strike direction in km
#            ld_hyp:                 distance of hypocenter to upper left corner of tha fault plane in dip direction in km
#            
#    Output: points:           local coordinates of each subfault (top-center of each subfault or segment: |'|)
#    ########################################
#   plot_fault_plane
#   This function plot slip on fault plane. The input is similar to mesh_fault_plane.
#   other input:    slip:   a vector containing slip of each subfault. The lenght of the vectro is n_strike*n_dip
# 
#    Written by Hadi Panahi (930905)
# #####################################################################################################################################################


def mesh_fault_plane(strike, dip, n_strike, n_dip, l_elem_strike, l_elem_dip, htop, ls_hyp, ld_hyp, flag ):
#    This function mesh the fault plane. The convention of the fucntion is due to Aki and Richard, 2nd edition,  Figure 4.20,  page 109
#    Thus X is in NS, y in EW and z positive in downward direction
#    The Coordinates are given for top-center of each subfault or segment: |'|
#    Origin of local coordinate system at hypocenter: X (NS) = 0,  Y (EW) = 0
#
#    For meshing the fault plane,  we consider two basis vectors, one in strike direction and other in dip direction
#    v_strike=(cos(fi), sin(fi), 0)
#    v_dip=(-cos(delta)sin(fi), cos(delta)cos(fi), sin(delta))   positive in downward direction
#    
#    Input:  strike:     strike angle in degree  =fi
#            dip:                        dip angle in degree  = delta
#            n_strike:                number of element in strike direction
#            n_dip:                    number of element in dip direction
#            len_elem_strike:    length of elements in strike direction in km
#            len_elem_dip:       length of elements in dip direction in km
#            htop:                     length to top of the fault plane in km
#            ls_hyp:                  distance of hypocenter to upper left corner of the fault plane in strike direction in km
#            ld_hyp:                 distance of hypocenter to upper left corner of tha fault plane in dip direction in km
#            flag:                      0: output is top-center of each subfault or segment: |'|
#                                       1: output is verteces of each subfault (without duplication)
#            
#    Output: points:           local coordinates of each subfault , for flag=0 the length of the output is (n_strike x n_dip) and for 
#                                                flag=1 the length of the output is ((n_strike+1) x (n_dip+1))
#            points, elem      elem: element tabel, coordinates of four vertices (1:lower left, 2:lower right, 3:upper right, 4:upper left) (941106)
#
#    Written by Hadi Panahi (930905)
#    Modified by Hadi Panahi (940508)       addition of flag
#    Modified by Hadi Panahi (941014)       addition of n_dip+1 for flag=1
#    Modified by Hadi Panahi (941106)       addition of flag=2 for element tabel
# #####################################################################################################################################################


    fi=np.radians(strike)
    delta=np.radians(dip)
    v_s=np.array([np.cos(fi), np.sin(fi), 0])
    v_d=np.array([-1.0*np.cos(delta)*np.sin(fi), np.cos(delta)*np.cos(fi), np.sin(delta)])
    

    ind=-1
    
    dd_hyp=ls_hyp*v_s+ld_hyp*v_d
    dd_hyp[2]=0.0                   # there is no transformation in z direction
    
    if flag==0:
        sc=0.5                  # 0.5 is because the coordinate is for midpoint of subfault
    elif ((flag==1) or (flag==2)):
        sc=0.0
        n_strike+=1
        n_dip+=1

    points=np.zeros([n_strike*n_dip, 3])

    for i_d in np.arange(0, n_dip)*l_elem_dip:
        for i_s in np.arange(0, n_strike)*l_elem_strike:
            ind+=1
            dd=np.zeros([3, ])
            dd=(i_s+sc*l_elem_strike)*v_s+i_d*v_d-dd_hyp    
            dd[2]=dd[2]+htop
            points[ind, :]=dd
   
    if flag==2:

        n_strike-=1
        n_dip-=1

        elem=np.zeros((n_strike*n_dip,4),dtype=np.uint16)
        ind_elem=-1

        for i_d in range(n_dip):
            A_dip=i_d*(n_strike+1)+1
            for i_s in range(n_strike):
                ind_elem+=1
                elem[ind_elem,3]=A_dip+i_s
                elem[ind_elem,2]=A_dip+i_s+1
                elem[ind_elem,1]=A_dip+(n_strike+1)+i_s+1
                elem[ind_elem,0]=A_dip+(n_strike+1)+i_s


        # Checking of elements
        epsi=4.4   # meter, tolerance
        elem_dist=np.array([l_elem_dip,l_elem_strike,l_elem_dip,l_elem_strike])
        vert=range(-1,4)
        dd=np.zeros(4)
        for i_el,el in enumerate(elem):
            for i in range(4):
                dd[i]=ddist(points[el[vert[i]]-1,:],points[el[vert[i+1]]-1,:])
            chs=np.sum(dd/elem_dist)
            if ((chs>4.04) or (chs<3.96)):
                txt='Element '+str(i_el)+' has problem'
            
        print
        txt='Checking of Elements has been ended !!!'
        print colored(txt,'red',attrs=['bold'])



    if (flag==0) or (flag==1):
        return points
    elif flag==2:
        return points,elem


# ###################################################################

def plot_fault_plane(strike, dip, n_strike, n_dip, l_elem_strike, l_elem_dip, htop, ls_hyp, ld_hyp, slip ):
    fi=np.radians(strike)
    delta=np.radians(dip)
    v_s=np.array([np.cos(fi), np.sin(fi), 0])
    v_d=np.array([-1.0*np.cos(delta)*np.sin(fi), np.cos(delta)*np.cos(fi), np.sin(delta)])
    
    points=np.zeros([(n_strike+1)*(n_dip+1), 3])
    ind=-1
    
    dd_hyp=ls_hyp*v_s+ld_hyp*v_d
    dd_hyp[2]=0.0                   # there is no transformation in z direction
    
    for i_d in np.arange(0, n_dip+1)*l_elem_dip:
        for i_s in np.arange(0, n_strike+1)*l_elem_strike:
            ind+=1
            dd=np.zeros([3, ])
            dd=i_s*v_s+i_d*v_d-dd_hyp
            dd[2]=dd[2]+htop
            points[ind, :]=dd
            
    X=np.reshape(points[:,0],(n_dip+1,n_strike+1))
    Y=np.reshape(points[:,1],(n_dip+1,n_strike+1))
    Z=-1*np.reshape(points[:,2],(n_dip+1,n_strike+1))
    
    cmp=[[1.000,1.000,1.000],[0.858,0.955,0.955],[0.715,0.910,0.910],[0.573,0.865,0.865],[0.430,0.820,0.820], \
    [0.288,0.775,0.775],[0.145,0.729,0.729],[0.135,0.747,0.681],[0.126,0.765,0.632],[0.116,0.784,0.584],\
    [0.106,0.802,0.535],[0.097,0.820,0.486],[0.087,0.838,0.438],[0.077,0.856,0.389],[0.068,0.874,0.340],\
    [0.058,0.892,0.292],[0.048,0.910,0.243],[0.039,0.928,0.195],[0.029,0.946,0.146],[0.019,0.964,0.097],\
    [0.010,0.982,0.049],[0.000,1.000,0.000],[0.056,1.000,0.000],[0.111,1.000,0.000],[0.167,1.000,0.000],\
    [0.222,1.000,0.000],[0.278,1.000,0.000],[0.333,1.000,0.000],[0.389,1.000,0.000],[0.444,1.000,0.000],\
    [0.500,1.000,0.000],[0.556,1.000,0.000],[0.611,1.000,0.000],[0.667,1.000,0.000],[0.722,1.000,0.000],\
    [0.778,1.000,0.000],[0.833,1.000,0.000],[0.889,1.000,0.000],[0.944,1.000,0.000],[1.000,1.000,0.000],\
    [1.000,0.923,0.000],[1.000,0.846,0.000],[1.000,0.769,0.000],[1.000,0.692,0.000],[1.000,0.615,0.000],\
    [1.000,0.538,0.000],[1.000,0.462,0.000],[1.000,0.385,0.000],[1.000,0.308,0.000],[1.000,0.231,0.000],\
    [1.000,0.154,0.000],[1.000,0.077,0.000],[1.000,0.000,0.000],[0.955,0.000,0.000],[0.909,0.000,0.000],\
    [0.864,0.000,0.000],[0.818,0.000,0.000],[0.773,0.000,0.000],[0.727,0.000,0.000],[0.682,0.000,0.000],\
    [0.636,0.000,0.000],[0.591,0.000,0.000],[0.545,0.000,0.000],[0.500,0.000,0.000]]
    
#    cmp=np.loadtxt('ff_color_list2.dat')
    c_map = mpl.colors.ListedColormap(cmp)
    ss=np.reshape(slip, (n_dip, n_strike))
    ss = np.hstack((ss, np.zeros((ss.shape[0], 1), dtype=ss.dtype)))
    ss = np.vstack((ss, np.zeros((1, ss.shape[1]), dtype=ss.dtype)))
    ss2=ss/ss.max()
    
     
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
 

    surf=ax.plot_surface(Y, X, Z, rstride=1, cstride=1,linewidth=0.5,vmin = np.min(slip), vmax = np.max(slip),  facecolors=c_map(ss2), shade=False)
    surf.set_edgecolor('k')

    ax.set_ylabel('X==NS [km]')
    ax.set_xlabel('Y==EW [km]')
    ax.set_zlabel('Depth [km]')
    ax.view_init(elev=40., azim=-1*strike-15)
#    ax.view_init(elev=90., azim=90)

    m = cm.ScalarMappable(cmap=c_map)
    m.set_array(ss)
    cbar=plt.colorbar(m)
    cbar.ax.set_ylabel('Slip [m]', rotation=90, labelpad=15)

    plt.show()

def plot_fault_staions(strike, dip,l_strike, l_dip, epi, stations ):
    # note that in this fucntion y is in North direction
    fig, ax = plt.subplots()
    plt.scatter(stations[:, 1], stations[:, 0], marker='^')
    plt.scatter(0, 0, marker='*', s=150, edgecolor='r', facecolor='r')
    for i in range(len(stations)):
        txt=str(i+1)
        ax.annotate(txt, (stations[i, 1], stations[i, 0]))
  
    plt.show()
    
def determine_moment(axi_data_file, fault_plane_file, u, l_elem_strike, l_elem_dip ):
        # ###################################################### Reading Axi.data
    f_dat=open(axi_data_file,'r')
    line=f_dat.readline()
    line=f_dat.readline()
    line=line.split(',')
    nc=int(line[0].strip()[3:])                                     # number of layers
    nfreq_str=line[1].strip()
    tl_str=line[2].strip()
    nfreq=float(nfreq_str[6:len(nfreq_str)])        # number of frequencies
    tl=float(tl_str[3:len(tl_str)])                           # tl=total duration in sec

    nr=int(line[4].strip()[3:])

    line=f_dat.readline()
    line=f_dat.readline()

    # #################### reading layers properties
    hc=np.zeros(nc)
    vp=np.zeros(nc)
    vs=np.zeros(nc)
    ro=np.zeros(nc)
    Qp=np.zeros(nc)
    Qs=np.zeros(nc)

    for ic in range(nc):
        line=f_dat.readline()
        line=line.split()

        hc[ic]=float(line[0])
        vp[ic]=float(line[1])
        vs[ic]=float(line[2])
        ro[ic]=float(line[3])
        Qp[ic]=float(line[4])
        Qs[ic]=float(line[5])

    if hc[0]==0.0:
        for ic in range(nc-1):
            hc[ic]=hc[ic+1]-hc[ic]

    f_dat.close()
    xmm=np.log(nfreq)/np.log(2.)
    mm=int(xmm)+1
    nt=2**mm
    dfreq=1./tl
    dt=tl/nt

    # ###################################################### Reading input date for sources
    xs=np.array([])
    ys=np.array([])
    zs=np.array([])
    ss=np.array([])
    ds=np.array([])
    rup=np.array([])
    mu=np.array([])
    isc=[]

    ind=-1

    source=np.loadtxt(fault_plane_file)
    ns=len(source)      # number of sources
    
    xmoment=0.0
    surf=l_elem_strike*l_elem_dip
    for iis in range(len(source)):
        hh=0.
        isc.extend([0])                         # these lines determines that the layer of each source
        zsc=source[iis, 2]*1000.
        for ic in range(nc-1):
            hh=hc[ic]
            if zsc>hh:
                zsc=zsc-hh
                isc[iis]=ic+1
            else:
                break

        mu=vs[isc[iis]]**2*ro[isc[iis]]
        xmoment+=mu*u[iis]*surf

    Mw=2./3.*np.log10(xmoment*1.0e7)-10.7
    return xmoment, Mw
 
# #######################################################################
# #######################################################################
# #######################################################################
# #######################################################################
# #######################################################################
 
# This function convert lat long to X Y using lambert  conformal conic projection based on USGS Bulletin no.1532, 1982, p. 105
# Input:        lat_r: lat of receiver
#               lon_r: lon of receiver
#               lat_s: lat of source
#               lon_s: lon of source
#               lat_ep: lat of epicenter
#               lon_ep: lon of epicenter
#
# Output:         x_r,y_r:  x , y of reciever in KM
#                 x_s,y_s: x , y of source in KM
#
#    Origin of local coordinate system at epicenter: X (NS) = 0,  Y (EW) = 0
#    The convention of the fucntion is due to Aki and Richard, 2nd edition,  Figure 4.20,  page 109
#    Thus x is increasing  northerly,  y is increasing easterly, and z positive in downward direction
#
#   Written by Hadi Panahi (940526)
# #######################################################################

def ll2km(lat_r, lon_r, lat_s, lon_s, lat_ep, lon_ep):

    lat_r=np.pi/180.*lat_r
    lon_r=np.pi/180.*lon_r
    lat_s=np.pi/180.*lat_s
    lon_s=np.pi/180.*lon_s
    lat_ep=np.pi/180.*lat_ep
    lon_ep=np.pi/180.*lon_ep
    
    # initialize lambert  conformal conic projection, the notation is as USGS Bulletin no.1532, 1982, p. 105, 'f' for fi and 'l' for lambda
    R=6378.0      #radius of earth
    # origin of rectangular coordinate, equal to epicenter coordinates
    f0=lat_ep
    l0=lon_ep
    
    # standard parallels
    f=np.append(lat_r, lat_s)
    f1=np.amin(f)
    f2=np.amax(f)
    
    # parameters of transformation
    n=np.log(np.cos(f1)/np.cos(f2))/np.log(np.tan(np.pi/4.+f2/2.)/np.tan(np.pi/4.+f1/2.))
    F=(np.cos(f1)*(np.tan(np.pi/4+f1/2.))**n)/n
    r0=R*F/(np.tan(np.pi/4+f0/2.))**n
    
    # performing transformation for receiver
    # f=lat_r     l=lon_r
    r=R*F/(np.tan(np.pi/4+lat_r/2.))**n
    teta=n*(lon_r-l0)
    
    y_r=r*np.sin(teta)                  # y is increasing easterly
    x_r=r0-r*np.cos(teta)            # x is increasing  northerly
    
    # performing transformation for source
    # f=lat_r     l=lon_r
    r=R*F/(np.tan(np.pi/4+lat_s/2.))**n
    teta=n*(lon_s-l0)
    
    y_s=r*np.sin(teta)                  # y is increasing easterly
    x_s=r0-r*np.cos(teta)            # x is increasing  northerly
    
    return x_r, y_r, x_s, y_s

    
    
   
# #######################################################################
# #######################################################################
# #######################################################################
# #######################################################################
# #######################################################################
 
# This function make a 2d interpolation withi linear functions based on JOURNAL OF GEOPHYSICAL RESEARCH, VOL. 109
# Note that the global element shall be rectangular
# Input:    xi,yi   coordinates of four vertices (1:lower left, 2:lower right, 3:upper right, 4:upper left)
#           ui      quantity of interest correspondig to different vertices
#           x,y     coordinate of desired point
#
# output:   u at desired point
# Written by Hadi Panahi (941106)
# #######################################################################

def interp_2d(xi,yi,ui,x,y):
    a=(abs(xi[0]-xi[1]))/2.0
    b=(abs(yi[0]-yi[3]))/2.0

    xc=(xi[0]+xi[1])/2.0
    yc=(yi[0]+yi[3])/2.0

    kis=(x-xc)/a
    eta=(y-yc)/b

    N=np.zeros(4)
    N[0]=0.25*(1-kis)*(1-eta)
    N[1]=0.25*(1+kis)*(1-eta)
    N[2]=0.25*(1+kis)*(1+eta)
    N[3]=0.25*(1-kis)*(1+eta)

    u=0.0
    for i in range(4):
        u+=N[i]*ui[i]


    return u

# #######################################################################
# #######################################################################
# #######################################################################
# #######################################################################
# #######################################################################
 
# This function make a 2d interpolation of green functions (vectors) with linear functions based on JOURNAL OF GEOPHYSICAL RESEARCH, VOL. 109
# Note that the global element shall be rectangular
# Input:    xi,yi   coordinates of four vertices (1:lower left, 2:lower right, 3:upper right, 4:upper left)
#           u1,u2,u3,u4:      green functions of different nodes with above order
#           x,y     coordinate of desired point
#
# output:   green function at desired point
#
# Written by Hadi Panahi/ Checked by Anooshiravan Ansari (941212)
# #######################################################################

def interp_green_2d(xi,yi,u1,u2,u3,u4,x,y):

    a=(abs(xi[0]-xi[1]))/2.0
    b=(abs(yi[0]-yi[3]))/2.0

    xc=(xi[0]+xi[1])/2.0
    yc=(yi[0]+yi[3])/2.0

    kis=(x-xc)/a
    eta=(y-yc)/b

    N=np.zeros(4)
    N[0]=0.25*(1-kis)*(1-eta)
    N[1]=0.25*(1+kis)*(1-eta)
    N[2]=0.25*(1+kis)*(1+eta)
    N[3]=0.25*(1-kis)*(1+eta)

    u=np.zeros(u1.shape)
    (m,n)=u1.shape
    ui=np.zeros((4,1))
    for i in range(m):
        for j in range(n):
            
            ui[0,0]=u1[i,j]
            ui[1,0]=u2[i,j]
            ui[2,0]=u3[i,j]
            ui[3,0]=u4[i,j]         
            for k in range(4):
                u[i,j]+=N[k]*ui[k]
                
    return u


# #######################################################################
# #######################################################################
# #######################################################################
# #######################################################################
# #######################################################################

def ddist(p1,p2):
    dd=((p1[0]-p2[0])**2.0+(p1[1]-p2[1])**2.0+(p1[2]-p2[2])**2.0)**0.5
    return dd



# #######################################################################
# #######################################################################
# #######################################################################
# #######################################################################
# #######################################################################
# This function mesh an element be specifying kis and eta of meshpoints inside the element
# This is useful in meshing an element in such a form that no node placed on the sides of the element
#
# Input:    xx           coordinates of four vertices (1:lower left, 2:lower right, 3:upper right, 4:upper left)
#                               (an nparray of n*2 dimentsion)
#           kis_eta      coordinate of desired points in kisi eta coordinate (an nparray of n*2 dimentsion)
#
# output:   x_elem       coordinate of desired points (an nparray of n*2 dimentsion ) in global coordinate system
#
# Written by Hadi Panahi (941107)
# #######################################################################

def mesh_subfault(xx,kis_eta):

    x_elem=np.zeros(kis_eta.shape)

    for i in range(len(kis_eta)):
        kis=kis_eta[i,0]
        eta=kis_eta[i,1]

        N=np.zeros(4)
        N[0]=0.25*(1-kis)*(1-eta)
        N[1]=0.25*(1+kis)*(1-eta)
        N[2]=0.25*(1+kis)*(1+eta)
        N[3]=0.25*(1-kis)*(1+eta)

        for j in range(4):
            x_elem[i,0]+=N[j]*xx[j,0]
            x_elem[i,1]+=N[j]*xx[j,1]

    return x_elem



# #######################################################################
# #######################################################################
# #######################################################################
# #######################################################################
# #######################################################################
# This function mesh an element in a mapped way
# This is useful in meshing an element in such a form that no node placed on the sides of the element
#
# Input:    xx           coordinates of four vertices (1:lower left, 2:lower right, 3:upper right, 4:upper left)
#                               (an nparray of n*2 dimentsion)
#           m,n          number of nodes in strike and dip directions respectivly
#
# output:   x_elem       coordinate of desired points (an nparray of n*2 dimentsion ) in global coordinate system
#
# Written by Hadi Panahi/ Checked by Anooshiravan Ansari (941117)
# #######################################################################

def mesh_subfault_mapped(xx,m,n):

    kis_eta=np.zeros((m*n,2))
    for i in range(0,m):
        for j in range(0,n):
            kis_eta[n*i+j,1]=1-1.0/m-2*i*(1.0/m)
            kis_eta[n*i+j,0]=-1+1.0/n+2*j*(1.0/n)


    x_elem=mesh_subfault(xx,kis_eta)
    return x_elem


# #######################################################################
# #######################################################################
# #######################################################################
# #######################################################################
# #######################################################################

#
# This function reads the parameters file and put the amount of each parameter in an ndarray in the below sequence
#
# Input:  filename:  the name of input file, in each row of input file, after each of the following identifier, the number 
#                    shall be presented after "=", eg: ep_lat=65.32
#################################################################################################################
## order  param id.         Description
## ------------------------------------------------------
## 0      ep_lat              latitude of hypocenter
## 1      ep_lon              longitude of hypocenter
## 2      dept                depth of hypocenter
## 3      strike:             strike angle in degree
## 4      dip:                dip angle in degree 
## 5      L:                  lenght of fault plane
## 6      W:                  width of fault plane
## 7      h_top:              distance to the top of fault in km
## 8      ls_hyp:             distance of hypocenter to upper left corner of the fault plane in strike direction in km
## 9      ld_hyp:             distance of hypocenter to upper left corner of tha fault plane in dip direction in km
## 10     n_strike_param:     number of element in strike direction
## 11     n_dip_param:        number of element in dip direction
## 12     div_strike_green:   number of subelement in each element in strike direction
## 13     div_dip_green:      number of subelement in each element in dip direction
## 14     div_strike_conv:    number of points in each subelement in strike direction for convolving
## 15     div_dip_conv:       number of points in each subelement in dip direction for convolving
## 16     dt:                 period of sampling
## 17     f_min:              minimum of frequency band uses in the inversion
## 18     f_max:              maximum of frequency band uses in the inversion
## 19     n_butter:           order of butterworth filter
##
#################################################################################################################
# 
# Output:  params:      an ndarray containing the input parameters in the order of above
#          err_code:    a list containing the code of the parameter with errer, err_code=order+1
#
# Written by Hadi Panahi/ Checked by Anooshiravan Ansari (941212)
# #######################################################################


def read_params(filename):


    file=open(filename,'r')
    with open(filename) as f:
        a=sum(1 for _ in f)
    params=np.ones((20,1))
    params*=-10000
    for i in range(a):
        line=file.readline()
        line=line.split('=')
   
        if line[0]=='ep_lat':
            b=is_number(line[1])
            if b==1:
                params[0,0]=np.float(line[1])
        elif line[0]=='ep_lon':
            b=is_number(line[1])
            if b==1:
                params[1,0]=np.float(line[1])
        elif line[0]=='depth':
            b=is_number(line[1])
            if b==1:
                params[2,0]=np.float(line[1])
        elif line[0]=='strike':
            b=is_number(line[1])
            if b==1:
                params[3,0]=np.float(line[1])
        elif line[0]=='dip':
            b=is_number(line[1])
            if b==1:
                params[4,0]=np.float(line[1])
        elif line[0]=='L':
            b=is_number(line[1])
            if b==1:
                params[5,0]=np.float(line[1])
        elif line[0]=='W':
            b=is_number(line[1])
            if b==1:
                params[6,0]=np.float(line[1])
        elif line[0]=='htop':
            b=is_number(line[1])
            if b==1:
                params[7,0]=np.float(line[1])
        elif line[0]=='ls_hyp':
            b=is_number(line[1])
            if b==1:
                params[8,0]=np.float(line[1])
        elif line[0]=='ld_hyp':
            b=is_number(line[1])
            if b==1:
                params[9,0]=np.float(line[1])
        elif line[0]=='n_strike_param':
            b=is_number(line[1])
            if b==1:
                params[10,0]=np.float(line[1])
        elif line[0]=='n_dip_param':
            b=is_number(line[1])
            if b==1:
                params[11,0]=np.float(line[1])
        elif line[0]=='div_strike_green':
            b=is_number(line[1])
            if b==1:
                params[12,0]=np.float(line[1])
        elif line[0]=='div_dip_green':
            b=is_number(line[1])
            if b==1:
                params[13,0]=np.float(line[1])
        elif line[0]=='div_strike_conv':
            b=is_number(line[1])
            if b==1:
                params[14,0]=np.float(line[1])
        elif line[0]=='div_dip_conv':
            b=is_number(line[1])
            if b==1:
                params[15,0]=np.float(line[1])
        elif line[0]=='dt':
            b=is_number(line[1])
            if b==1:
                params[16,0]=np.float(line[1])
        elif line[0]=='f_min':
            b=is_number(line[1])
            if b==1:
                params[17,0]=np.float(line[1])
        elif line[0]=='f_max':
            b=is_number(line[1])
            if b==1:
                params[18,0]=np.float(line[1])
        elif line[0]=='n_butter':
            b=is_number(line[1])
            if b==1:
                params[19,0]=np.float(line[1])

# This section is for checking that all the inputs are correct
# if there is any error, it was displayed and an error code is set
    err_code=[0]
    if params[0,0]==-10000.0:
        print colored('ep_lat is not defined!!!','red',attrs=['bold'])
        err_code.append(1)
    if params[1,0]==-10000.0:
        print colored('ep_lon is not defined!!!','red',attrs=['bold'])
        err_code.append(2)
    if params[2,0]==-10000.0:
        print colored('depth is not defined!!!','red',attrs=['bold'])
        err_code.append(3)
    if params[3,0]==-10000.0:
        print colored('strike is not defined!!!','red',attrs=['bold'])
        err_code.append(4)
    if params[4,0]==-10000.0:
        print colored('dip is not defined!!!','red',attrs=['bold'])
        err_code.append(5)
    if params[5,0]==-10000.0:
        print colored('L is not defined!!!','red',attrs=['bold'])
        err_code.append(6)
    if params[6,0]==-10000.0:
        print colored('W is not defined!!!','red',attrs=['bold'])
        err_code.append(7)
    if params[7,0]==-10000.0:
        print colored('htop is not defined!!!','red',attrs=['bold'])
        err_code.append(8)
    if params[8,0]==-10000.0:
        print colored('ls_hyp is not defined!!!','red',attrs=['bold'])
        err_code.append(9)
    if params[9,0]==-10000.0:
        print colored('ld_hyp is not defined!!!','red',attrs=['bold'])
        err_code.append(10)
    if params[10,0]==-10000.0:
        print colored('n_strike_param is not defined!!!','red',attrs=['bold'])
        err_code.append(11)
    if params[11,0]==-10000.0:
        print colored('n_dip_param is not defined!!!','red',attrs=['bold'])
        err_code.append(12)
    if params[12,0]==-10000.0:
        print colored('div_strike_green is not defined!!!','red',attrs=['bold'])
        err_code.append(13)
    if params[13,0]==-10000.0:
        print colored('div_dip_green is not defined!!!','red',attrs=['bold'])
        err_code.append(14)
    if params[14,0]==-10000.0:
        print colored('div_strike_conv is not defined!!!','red',attrs=['bold'])
        err_code.append(15)
    if params[15,0]==-10000.0:
        print colored('div_dip_conv is not defined!!!','red',attrs=['bold'])
        err_code.append(16)
    if params[16,0]==-10000.0:
        print colored('dt is not defined!!!','red',attrs=['bold'])
        err_code.append(17)
    if params[17,0]==-10000.0:
        print colored('f_min is not defined!!!','red',attrs=['bold'])
        err_code.append(18)
    if params[18,0]==-10000.0:
        print colored('f_max is not defined!!!','red',attrs=['bold'])
        err_code.append(19)
    if params[19,0]==-10000.0:
        print colored('n_butter is not defined!!!','red',attrs=['bold'])       
        err_code.append(20)
        
    return params,err_code
        
def is_number(s):
    try:
        float(s)
        return 1
    except ValueError:
        return 0



###########################################################################
## this function takes the parameters of the fault and get 5 files that each contains a kind of information we needs for furthur works.
## input:
##       1. file_identifier: this is an identifier for specifying the fault
##       2. params_file: this text file contains all characteristics of the faullt we need and the information in it are as 
##           order that defined in read_params function in ans_finite_fault_tools.py
##       3. save_flag: if save_flag=1 output will save in text files, if save_flag=0 generated data just save in 5 array
##       4. plot: if plot=1 the faults with all node that we need their information plot in 3d view
## ---------------------------------------------------------------------------------------------
## output: in meshing fault we have 3 kind of nodes and 2 kind of elements. first element is subfault and we call them elem_param
##         and this element defines with 4 nodes that shape it in this order (1:lower left, 2:lower right, 3:upper right, 4:upper left)
##         and this nodes are param_nodes. each subfault convert to finer element that called green_elem and the nodes define these 
##         elements are green_nodes. in each green_element n*m nodes with function of mesh_subfault_mapped in ans_finite_fault.py defines
##         and this are conv_nodes.
##  ------------------------------------------------
##       1. coord_param: contain index and coordinates of param_nodes. its 4 columns are index,x,y and z.
##       2. elem_param: contains index and number of param_nodes that make this elemnt. nodes number are 
##          in order (1:lower left, 2:lower right, 3:upper right, 4:upper left)
##       3. coord_green_points: contain index and coordinates of green_points. its 4 columns are index,x,y and z.
##       4. elem_green_points:  contain number of the green_points that create this elements and the number of param_nodes 
##          that make the elem_param that this elem_green is in it.
##       5. coord_elem_conv: contains the coordinates of each conv_nodes and number of nodes of elem_green and elem_param that are
##          in them.


def mesh_fault(file_identifier,params_file,save_flag,plot):

    [params,err_code]=read_params(params_file)

    ep_lat=params[0,0]
    ep_lon=params[1,0]
    depth=params[2,0]
    strike=params[3,0]
    dip=params[4,0]
    L=params[5,0]
    W=params[6,0]
    htop=params[7,0]
    ls_hyp=params[8,0]
    ld_hyp=params[9,0]
    n_strike_param=int(params[10,0])
    n_dip_param=int(params[11,0])
    div_strike_green=int(params[12,0])
    div_dip_green=int(params[13,0])
    div_strike_conv=int(params[14,0])
    div_dip_conv=int(params[15,0])

    l_elem_strike=L/n_strike_param
    l_elem_dip=W/n_dip_param   
    
#------------------------------------------------------------------------------#
#---- 1th file: coord_param
    s_p=mesh_fault_plane(strike, dip, n_strike_param, n_dip_param, l_elem_strike, l_elem_dip, htop, ls_hyp, ld_hyp, 1 )

    coord_param=np.zeros((len(s_p),4))        
    for i,s in enumerate(s_p):
        coord_param[i,0]=int(i+1)
        coord_param[i,1]=s[0]*1000
        coord_param[i,2]=s[1]*1000
        coord_param[i,3]=s[2]*1000
#---- 2th file: elem_param
    elem_param=np.zeros(((n_strike_param*n_dip_param),5))               
    for i in range(n_strike_param*n_dip_param):
        ii=(i//n_strike_param)
            
        elem_param[i,0]=int(i+1)
        elem_param[i,1]=i+ii+n_strike_param+2
        elem_param[i,2]=i+ii+n_strike_param+3
        elem_param[i,3]=i+ii+2
        elem_param[i,4]=i+ii+1
#---- 3th file: coord_green_points           
    l_elem_strike=L/(n_strike_param*div_strike_green) 
    l_elem_dip=W/(n_dip_param*div_dip_green)  


    s_p_green=mesh_fault_plane(strike, dip,n_strike_param*div_strike_green, n_dip_param*div_dip_green, l_elem_strike, l_elem_dip, htop, ls_hyp, ld_hyp, 1 )

    coord_green_points=np.zeros((len(s_p_green),4))    
    for i,s in enumerate(s_p_green): 
        coord_green_points[i,0]=int(i+1)
        coord_green_points[i,1]=s[0]*1000
        coord_green_points[i,2]=s[1]*1000
        coord_green_points[i,3]=s[2]*1000            

#---- 4th file: elem_green_points          
    elem_green_points=np.zeros(((n_strike_param*div_strike_green*n_dip_param*div_dip_green),9))
    for i in range(n_strike_param*div_strike_green*n_dip_param*div_dip_green):
        a=(i)%(n_strike_param*div_strike_green)+1 # define green column
        b=(a//div_strike_green)+1 # define param column
        if a%div_strike_green==0:
            b=b-1
        
        c=(i)//(n_strike_param*div_strike_green*div_dip_green)+1 # defime param row
    
        n=(c-1)*n_strike_param+b # defien number of param square
    
        nn=((n-1)//n_strike_param) # number of param row -1
    
        ii=(i//div_strike_green)/n_strike_param # number of green row -1
                      
        elem_green_points[i,0]=int(i+1)
        elem_green_points[i,1]=i+ii+n_strike_param*div_strike_green+2
        elem_green_points[i,2]=i+ii+n_strike_param*div_strike_green+3
        elem_green_points[i,3]=i+ii+2
        elem_green_points[i,4]=i+ii+1
        elem_green_points[i,5]=n+nn+n_strike_param+1
        elem_green_points[i,6]=n+nn+n_strike_param+2
        elem_green_points[i,7]=n+nn+1
        elem_green_points[i,8]=n+nn
            
#---- 5th file: coord_elem_conv          
    coord_elem_conv=np.zeros(((div_strike_conv*div_dip_conv*len(elem_green_points),12)))
    p=np.zeros((div_strike_conv*div_dip_conv*len(elem_green_points),3))        
    for j in range(len(elem_green_points)):
    
        s1=elem_green_points[j,1]
        s2=elem_green_points[j,2]
        s3=elem_green_points[j,3]
        s4=elem_green_points[j,4]

        ss1=elem_green_points[j,5]
        ss2=elem_green_points[j,6]
        ss3=elem_green_points[j,7]
        ss4=elem_green_points[j,8]

        x=np.zeros((4,3))
        x[0,:]=coord_green_points[s1-1,1:4]
        x[1,:]=coord_green_points[s2-1,1:4]
        x[2,:]=coord_green_points[s3-1,1:4]
        x[3,:]=coord_green_points[s4-1,1:4]

        xx=np.zeros((4,2))
        xx[:,0]=x[:,0]
        xx[:,1]=x[:,1]
            
        xy=mesh_subfault_mapped(xx,div_strike_conv,div_dip_conv)
            
        xx[:,0]=x[:,0]
        xx[:,1]=x[:,2]
            
        xz=mesh_subfault_mapped(xx,div_strike_conv,div_dip_conv)
        xyz=np.zeros((div_strike_conv*div_dip_conv,3))
        xyz[:,0]=xy[:,0]
        xyz[:,1]=xy[:,1]
        xyz[:,2]=xz[:,1]
    
        for i,s in enumerate(xyz): 
                
            coord_elem_conv[i+j*(len(xyz)),0]=i+j*(len(xyz))+1
            coord_elem_conv[i+j*(len(xyz)),1]=s[0]
            coord_elem_conv[i+j*(len(xyz)),2]=s[1]
            coord_elem_conv[i+j*(len(xyz)),3]=s[2]
            coord_elem_conv[i+j*(len(xyz)),4]=s1
            coord_elem_conv[i+j*(len(xyz)),5]=s2
            coord_elem_conv[i+j*(len(xyz)),6]=s3
            coord_elem_conv[i+j*(len(xyz)),7]=s4
            coord_elem_conv[i+j*(len(xyz)),8]=ss1
            coord_elem_conv[i+j*(len(xyz)),9]=ss2
            coord_elem_conv[i+j*(len(xyz)),10]=ss3
            coord_elem_conv[i+j*(len(xyz)),11]=ss4
      
            p[i+j*(len(xyz)),0]=s[0]
            p[i+j*(len(xyz)),1]=s[1]
            p[i+j*(len(xyz)),2]=s[2]
##############################################################################
## saving data in text files
    if save_flag==1:
        
        np.savetxt(file_identifier+'_coord_param.txt',coord_param,fmt='%3.i %15.4f %15.4f %15.4f'  )
        np.savetxt(file_identifier+'_elem_param.txt',elem_param,fmt='%3.i %14.i %7.i %7.i %7.i'  )
             
        np.savetxt(file_identifier+'_coord_green_points.txt',coord_green_points,fmt='%3.i %15.4f %15.4f %15.4f'  )
        np.savetxt(file_identifier+'_elem_green_points.txt',elem_green_points,fmt='%3.i %14.i %7.i %7.i %7.i %14.i %7.i %7.i %7.i'  )

        np.savetxt(file_identifier+'_coord_elem_conv.txt',coord_elem_conv,fmt='%7.i %15.4f %15.4f %15.4f %14.i %7.i %7.i %7.i %14.i %7.i %7.i %7.i' )        

########################################################################################
#  ploting the meshed fault
    p=p/1000
    for i in range(len(s_p)):
        for j in range(3):
            if np.abs(s_p[i,j])<0.00001:
                s_p[i,j]=0

    for i in range(len(s_p_green)):
        for j in range(3):
            if np.abs(s_p_green[i,j])<0.00001:
                s_p_green[i,j]=0  

    for i in range(len(p)):
        for j in range(3):
            if np.abs(p[i,j])<0.00001:
                p[i,j]=0          

    
    if not plot:
        plot=0

    if plot==1:

        fig = plt.figure(1)
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(s_p[:,1], s_p[:,0], s_p[:,2], s=200,c='r', marker='*')

        ax.scatter(s_p_green[:,1], s_p_green[:,0], s_p_green[:,2],s=15, c='g', marker='^')
        
        ax.scatter(p[:,1], p[:,0], p[:,2],s=2, c='k', marker='o')
        for i in range(len(s_p)):
            ax.text(s_p[i,1], s_p[i,0], s_p[i,2],  '%s' % (str(i+1)), size=10, zorder=1,color='k') 
    
        ax.set_xlabel('EW (km)')
        ax.set_ylabel('NS (km)')
        ax.set_zlabel('Z (km)')
        plt.gca().invert_zaxis()

        plt.show()

    return coord_param,elem_param,coord_green_points,elem_green_points,coord_elem_conv

##########################################################################
## this function takes parameters of subfaults in SIV format. The SIV file contain coordinates os
## top_center of each subfault and its parameters. this function first check that our subfaults 
## are the same by comparing the coordiante of subfault in SIV format by coordinates of 2 top 
## nodes of each subfault. if there is subfault that does not match its number will be shown and 
## save in a file. then it calculate parameters of each nodes by parameters of surrounding subfault of that node.
## -------------------------------------------------------------------------
## inputs:
##       1. file_identifier: this is an identifier for specifying the fault
##       2. SIV_file: files that contain coordinates of top_center of each subfault 
##                    in 5 columns: lat , long, x , y and z. and its four parameters
##          skipr: skip number of rows at the begening of SIV files (header rows)
##       3. params_file: this text file contains all characteristics of the fault we need and the information in it
##                  are as order that defined in read_params function in ans_finite_fault_tools.py
##       4. elem_param: this file created by mesh_fault.py function in ans_finite_fault_tools.py
##       5. save_flag: if it is equal 1 the data will save in a text file.
## -------------------------------------------------------------------------
## output: 
##       1.parameter of param_points: contains 4 column that are parameters of each node. parameters 
##                                    are slip(m),rake,rise time and rupture time
##       2.err_code: a list containing the code of the subfaults that does not matches.
##
##  Writted by Hadi Panahi (950117) Chechked by A. Ansari
########################################################################## 
def param_nodes_params(identidier,SIV_file,skipr,params_file,elem_param,save_flag):
##  first part: take parameter of subfault and parameters for meshing
    f=np.loadtxt(SIV_file,skiprows=skipr)

    [params,err_code]=read_params(params_file)

    ep_lat=params[0,0]
    ep_lon=params[1,0]
    depth=params[2,0]
    strike=params[3,0]
    dip=params[4,0]
    L=params[5,0]
    W=params[6,0]
    htop=params[7,0]
    ls_hyp=params[8,0]
    ld_hyp=params[9,0]
    n_strike_param=params[10,0]
    n_dip_param=params[11,0]
    div_strike_green=params[12,0]
    div_dip_green=params[13,0]
    div_strike_conv=params[14,0]
    div_dip_conv=params[15,0]
    
    l_elem_strike=L/n_strike_param
    l_elem_dip=W/n_dip_param   
    
    s_p=mesh_fault_plane(strike, dip, n_strike_param, n_dip_param, l_elem_strike, l_elem_dip, htop, ls_hyp, ld_hyp, 1 )
###############################################################################
# second part: check the the coordinate of each two points of a subfault in strike direction with the coordinates of top-center of each subfault
# if any elemet does match with our data a message will show

#ff is the file with number of element and its parameters

#ff=np.savetxt('s_p.txt',s_p)
    (m,n)=f.shape
    ff=np.zeros((len(f),n+1))
    d=np.zeros((len(f),3))
    err_code=[0]
    for i in range(len(f)):
        d[i,0]=(s_p[i+i//n_strike_param,0]+s_p[i+i//n_strike_param+1,0])/2
        d[i,1]=(s_p[i+i//n_strike_param,1]+s_p[i+i//n_strike_param+1,1])/2
        d[i,2]=(s_p[i+i//n_strike_param,2]+s_p[i+i//n_strike_param+1,2])/2
        for j in range(len(f)):
            if abs(d[i,0]-f[j,3])<0.1 and abs(d[i,1]-f[j,2])<0.1 and abs(d[i,2]-f[j,4])<0.1 :  # increase 0.1 f[j,3] is for y ,f[j,4] is for x ,f[j,4] is for z
                ff[i,0]=i+1
                ff[i,1:]=f[i,:]

        if ff[i,0]==0:
            err_code.append(i+1)
        
###############################################################################
## third part: in this part  with help of Elem_param.txt file, it makes a matrix that in it spesifies each node is in contact with each elements     
################## stop the script if ...
    nodef=np.zeros((len(s_p),6))
    if type(elem_param)==type(identidier):
        elemf=np.loadtxt(elem_param)
    else:
        elemf=elem_param
    for i in range(len(s_p)):
        nodef[i,0]=i+1
    for j in range(len(elemf)):
        for k in range(4):
            nodef[elemf[j,k+1]-1,nodef[elemf[j,k+1]-1,5]+1]=j+1
            nodef[elemf[j,k+1]-1,5]=nodef[elemf[j,k+1]-1,5]+1


########################################
## it calculates each parameters of nodes by parameters of subfaults


    paramf=np.ndarray((len(nodef),4)) # we consider paramf for 5 parameters
#    for i in range(len(paramf)):
#        paramf[i,0]=np.int(i+1)

#    paramf[:,1]=s_p[:,0]
#    paramf[:,2]=s_p[:,1]
#    paramf[:,3]=s_p[:,2]


    for i in range(len(nodef)):
        a=0
        b=0
        c=0
        d=0
        for j in range(4):
        
            if nodef[i,j+1]!=0:
            
                 a+=f[nodef[i,j+1]-1,5]
                 b+=f[nodef[i,j+1]-1,6]
                 c+=f[nodef[i,j+1]-1,7]
                 d+=f[nodef[i,j+1]-1,8]

    
        paramf[i,0]=a/nodef[i,-1]
        paramf[i,1]=b/nodef[i,-1]
        paramf[i,2]=c/nodef[i,-1]
        paramf[i,3]=d/nodef[i,-1]

    if save_flag==1: 

        np.savetxt(identidier+'_parameters_of_param_points.txt',paramf,fmt=' %10.4f %10.4f %10.4f %10.4f' )

    return paramf,err_code




