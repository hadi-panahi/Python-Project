import numpy as np
import os
import glob
import sys, select
from shutil import move,copyfile
from scipy import signal
from ans_finite_fault_tools import *
import matplotlib.pyplot as plt
from termcolor import colored
import multiprocessing as mp
import timeit


def cal_green_par(axitra_dir,dir_name,s_pp,source_ind,process_no,cur,ns):
    # this function calculates green function for a portion of sources
    # it copies axitra and related files from 'axitra_dir' to a new dir_name 
    # and calculate the greens function for 's_pp' source points starting from 
    # index of 'source_ind'. This function is written for multiprocessing
    # all the green files will also be moved to 'dir_name', ns is total number
    # of source points and cur is number of points calculated

    print 'Process '+str(process_no)+' was started!'
# making the directory
    dir_name=os.path.normpath(dir_name) + os.sep     # adds '/' at the end of path name if it does not exist
    axitra_dir=os.path.normpath(axitra_dir) + os.sep


    ddir=dir_name+str(source_ind)+'/'
    if not os.path.exists(ddir):
        os.makedirs(ddir)

# copying files to the directory
 
    copyfile(axitra_dir+'axi.data',ddir+'axi.data')
    copyfile(axitra_dir+'axitra_ans',ddir+'axitra_ans')
    copyfile(axitra_dir+'station',ddir+'station')

    os.chdir(ddir)
    os.system("chmod +x axitra_ans")

    for s_ind,s in enumerate(s_pp):
        dum=cur.value
        dum+=1
        cur.value=dum

        txt=str(process_no)+'    '+str(source_ind+s_ind+1)+'    #'+str(cur.value)+'/'+str(ns)
        print txt
        f_source=open(ddir+'source','w')
        f_source.write("1 {0:10.5f} {1:10.5f} {2:10.5f} \n".format(s[0]*1000,s[1]*1000,s[2]*1000))
        f_source.close()

        f_inp=open(ddir+'axitra_ans.inp','w')
        txt=str(source_ind+s_ind+1)+'\n'
        f_inp.write(txt)
        f_inp.close()
        
        os.system("./axitra_ans < axitra_ans.inp")

        f_list=glob.glob('*.res')
        for fff in f_list:
            move(fff,dir_name)



# ################################################################################################################ the main program
if __name__ == '__main__':

    
    start_time_calc = timeit.default_timer()
    params_file='/home/hadi/Parkfield/00-Meshing/park_params.txt'
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
    dt=params[16,0]


    l_elem_strike=L/n_strike_param
    l_elem_dip=W/n_dip_param   


    n_strike=n_strike_param*div_strike_green
    n_dip=n_dip_param*div_dip_green

    l_elem_strike=L/n_strike 
    l_elem_dip=W/n_dip    

    st_file='/home/hadi/Parkfield/01-Green_Function/st_2.txt'
    layer_file='/home/hadi/Parkfield/Parkfield_velocity_model.txt'

    dest_green_dir='/home/hadi/Parkfield/01-Green_Function/Green_Func_res2/'
    axitra_dir='/home/hadi/Parkfield/01-Green_Function'

    # ########################################
    # ########## meshing the fault plane
    print colored('Meshing the fault plane','red',attrs=['bold'])
    print

    s_p=mesh_fault_plane(strike, dip, n_strike, n_dip, l_elem_strike, l_elem_dip, htop, ls_hyp, ld_hyp, 1 )
    # the s_p was verified by the SRCMOD

    f=open('source_total.txt','w')
    for i,s in enumerate(s_p):
        txt='{0:5d}  {1:15.2f}   {2:15.2f}    {3:15.2f} \r\n'.format(i+1,s[0]*1000,s[1]*1000,s[2]*1000)
        f.write(txt)
    f.close()

    txt='Total No. of source points='+str(len(s_p))
    print colored(txt,'green',attrs=['bold'])

    # ################ loading the stations' coordinates
    print colored('Loading the station coordinates and converting to xy','red',attrs=['bold'])
    print

    st_coord = np.loadtxt(st_file, usecols=[1,2])
    st_label = np.loadtxt(st_file, usecols=[0],dtype=np.str)

    x_r, y_r, x_s, y_s=ll2km(st_coord[:,0], st_coord[:,1], st_coord[0,0], st_coord[0,1], ep_lat, ep_lon) # the coordinates for source are dummy
    r_p=np.column_stack((x_r,y_r))
    nr=len(x_r)

    print colored('Creating station file ...','green',attrs=['bold'])
    print

    f=open('station','w')
    for i,r in enumerate(r_p):
        txt='{0:5d}  {1:15.2f}   {2:15.2f}      0.000 \r\n'.format(i,r[0]*1000,r[1]*1000)
        f.write(txt)
    f.close()

    txt='Total No. of Stations='+str(nr)
    print colored(txt,'green',attrs=['bold'])

    # ####### plotting fault plane and stations
    print
    out = raw_input("Plot the Source and Stations? <y/n[n]>: ")
    if not out:
        out='n'
    if out.lower()=='y':
        plt.figure(1)
        plt.plot(s_p[:,1],s_p[:,0],'bo',markeredgecolor='b')     # for plotting we replace x and y axes
        plt.plot(r_p[:,1],r_p[:,0],'k^')
        plt.plot(0,0,'r*',markersize=20.0)
        plt.xlabel('EW (km)')
        plt.ylabel('NS (km)')
        plt.grid(True)

        for i,txt in enumerate(st_label):
            plt.annotate(txt,(r_p[i,1],r_p[i,0]))

        plt.show()

    # #################### calculate maximum fault points to stations including ruputre length
    print
    print colored('Calculate maximum fault points to stations','red',attrs=['bold'])
    print
    d_max=0
    d_min=50
    for s in s_p:
        d_rup=(s[0]**2+s[1]**2)**0.5
        for r in r_p:
            d=((s[0]-r[0])**2.0+(s[1]-r[1])**2 )**0.5+d_rup
            if d>d_max:
                d_max=d
            if d<d_min:
                d_min=d
                

    print colored('--------------------------------------------------------','green',attrs=['bold'])
    txt1='Maximum distance (fault point+rupture dist)={0:6.2f}  km'.format(d_max)
    txt2='Minimum distance (fault point+rupture dist)={0:6.2f}  km'.format(d_min)
    print colored(txt1,'green',attrs=['bold'])
    print colored(txt2,'green',attrs=['bold'])
    
    # ################### calculate maximum trace length and compute the corresponding parameters of axi.data
    t_max=d_max/(0.9*2.5)+10       # it is assumed that the V_rup and V_reighl both are 0.9*Vs and Vs=2.5 km/s, an extra 10 sec is also added

    n_t=np.ceil(t_max/dt)
    nfreq = 1                      # determine nextpow2 
    while nfreq < n_t:
        nfreq *= 2

    tl=nfreq*dt

    txt='t_max={0:6.2f}  sec'.format(t_max)
    print colored(txt,'green',attrs=['bold'])

    txt='nfreq='+str(nfreq)
    print colored(txt,'green',attrs=['bold'])

    txt='dt='+str(dt)+' sec'
    print colored(txt,'green',attrs=['bold'])

    txt='tl='+str(tl)
    print colored(txt,'green',attrs=['bold'])
    print colored('--------------------------------------------------------','green',attrs=['bold'])
    # ################## reading the velocity structure
    print
    print colored('Loading velocity structure','red',attrs=['bold'])
    print

    layer=np.loadtxt(layer_file,skiprows=6)

    txt='No  thikness(m)  Vp(m/s)  Vs(m/s)     Density(kg/m3)     Qp     Qs'
    print colored(txt,'yellow',attrs=['bold'])
    txt='-----------------------------------------------------------------'
    print colored(txt,'yellow',attrs=['bold'])

    ind=0
    for rr in layer:
        ind+=1
        row=rr.astype(int)
        txt='{0:2d}  {1:8d}  {2:8d}  {3:8d}      {4:8d}        {5:5d}  {6:5d}'.format(ind,row[0],row[1],row[2],row[3],row[4],row[5])
        print colored(txt,'yellow',attrs=['bold'])     

    nc=ind
    # ##################################### writing axi.data
    print
    print colored('Writing axi.data ...','red',attrs=['bold'])

    f=open('axi.data','w')
    txt=' &input \r\n'
    f.write(txt)

    txt='nc={0:<2d},nfreq={1:<5d},tl={2:<6.2f},aw=1.0,nr={3:<3d},ns=1,xl=500000,ikmax=100000 \r\n'.format(int(nc),int(nfreq),tl,nr)
    f.write(txt)

    txt=' latlon=.false.,sourcefile="source",statfile="station" \r\n'
    f.write(txt)

    txt=' // \r\n'
    f.write(txt)

    for row in layer:
        txt='{0:<10.2f}  {1:8.2f}  {2:8.2f}  {3:8.2f} {4:8.2e} {5:8.2e} \r\n'.format(row[0],row[1],row[2],row[3],row[4],row[5])
        f.write(txt)

    f.close()

    print colored('File axi.data was created!!!','green',attrs=['bold'])
    print

    # ################################## calculation of Green functions
    n_sp_process=100
    cur=mp.Value('i', 0)

    grn=[]
    print
    out = raw_input("Do you want to calculate Green functions? <y/n[n]>: ")
    if not out:
        out='n'
    if out.lower()=='y':
        # determine no of procesess for parallel processing
        ns=len(s_p)
        n_process=int(ns//n_sp_process)
        s_ind_par=range(0,ns,n_sp_process)
        if ns%n_sp_process!=0:
            n_process+=1
            s_ind_par.append(ns)


        for n_proc in range(n_process):
            grnn=mp.Process(target=cal_green_par, args=(axitra_dir,dest_green_dir,s_p[s_ind_par[n_proc]:s_ind_par[n_proc+1]],s_ind_par[n_proc],n_proc,cur,ns))
            grn.append(grnn)
            grn[n_proc].start()

        for n_proc in range(n_process):
            grn[n_proc].join()



    stop_time_calc = timeit.default_timer()

    print stop_time_calc - start_time_calc 

 
