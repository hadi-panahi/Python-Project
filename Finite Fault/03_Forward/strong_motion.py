import numpy as np
from scipy.fftpack import fft
from obspy.core import *
import obspy.signal
#from v1py import *
from numpy import linalg as LA
from scipy.linalg import expm

# This file contains fucntions for processing of strong motion records

#  INTE:
#  [y]=inte(x,dt)
#
#  Description:  Obtain vector y by integrating vector x using trapezoidal rule
#
#               dt: Time Step (second)
#
# -------------------------------------------------------------------------------

def inte(x, dt):
    y=np.zeros(x.shape)
    dt2=dt/2
    y[0]=0
    for i in range(1, len(x)):
        y[i]=y[i-1]+(x[i-1]+x[i])*dt2

    return y

# #################################################################################################################
# #################################################################################################################
#  DIF:
#  [y]=dif(x,dt)
#
# Description:  Obtain vector y by differentiating vector x using 
#               central defference rule
#
#               dt: time step
#
# --------------------------------------------------------------------

def dif(x, dt):
    
    y=np.zeros(x.shape)
    npp=len(x)
    for i in range(1, npp-1):
        y[i]=x[i+1]-x[i-1]
    
    y[0]=(x[1]-x[0])*2
    y[npp-1]=(x[npp-1]-x[npp-2])*2
    y=y/(dt+dt)
    return y
   

# #################################################################################################$
# #################################################################################################$
#  ROT:
#  [x2,y2]=rot(x1,y1,alpha)
#  Description:  Rotate vectors x1 and y1 by degree alpha
#  (counter-clockwise) to vector x2 and y2
# -------------------------------------------------------

def rot(x1,y1,alpha):
    alpha=alpha*np.pi/180
    x2= x1*np.cos(alpha)+y1*np.sin(alpha)
    y2=-x1*np.sin(alpha)+y1*np.cos(alpha);

    return x2,y2


 
# #################################################################################################################
# #################################################################################################################

# This fucntion calculates fft spectrum of a signal sig with sampling period del_t and n_fft point for fft calculation    

def fft_spec(sig,del_t,n_fft):
    
    # Usage: [f,fft_sig1,fft_amp,fft_phase]=fft_spec(sig,del_t,n_fft)

    fft_sig=fft(sig,n_fft)
    f=np.arange(0,n_fft/2+1)*1.0/n_fft/del_t
    llen=len(f)

    fft_amp=np.abs(fft_sig[0:llen])
    fft_phase=np.angle(fft_sig[1:llen])
    fft_sig1=fft_sig[0:llen]
    
    return f,fft_sig1,fft_amp,fft_phase
  
# #################################################################################################################
# #################################################################################################################

# This fucntion smootthes the fft spectrum (or any other spectrum) with logarithmic scale

def smooth_fft_spec_new(fft_sig,rr,n_fft,dt,n):
    df=1.0/(dt*n_fft)
    Fn=df*n_fft/2.0

    a=np.logspace(np.log10(df),np.log10(Fn),n)
    len_a=len(a)

    ind=np.round(a/df)

    dum=np.round(a/df*rr)

    llen=len(fft_sig)
    fft_sig_s=np.zeros((len_a, 1))
    for i in range(len_a):
        fft_sig_s[i]=np.mean(fft_sig[np.max([ind[i]-dum[i],1]):np.min([ind[i]+dum[i]+1,llen])])
    

    return a,fft_sig_s

# #####################  baseline correction

def remove_baseline(x, y, tp, order):
    dt=x[1]-x[0]
    ind=int(tp/dt)
    xx=x[ind:len(x)]-x[ind]
    d=y[ind:len(y)]
        
    G=np.zeros((len(xx), 3))
    G[:, 0]=xx
    G[:, 1]=xx**2.0
    G[:, 2]=xx**3.0
    Y=[0]	
    GG=LA.inv(np.dot(G.T, G))
    m=np.dot(np.dot(GG, G.T), d)
    base=m[0]*xx+m[1]*xx**2.0+m[2]*xx**3.0
   
    yy=d-base
    y[ind:len(y)]=yy
    return y
    
# #################################################################################################################
# #################################################################################################################
# this function reads a v1 file and decimates all channels and shift the trace
# Inputs:   filname: filename of v1 file
#               n_down: an integer of downsampling (f_sample_new=f_sample/n_down)
#               f_filt: frequency of lowpass filter, should be less than 0.4xdownsampling frequency, otherwise it is set to this value
#               tp_obsr: time of observed p onset
#               tp_theory: time of theorithical p onset      (time shift=tp_theory-tp_obsr)
#               az_T: azimuth of T component
#
# Output    st_d: an obspy stream with three channels (E,V,N) respectively
#
# Written by AHadi Panahi (93.06.10)
# #################################################################

def decimate_shift_rotate(filname, n_down, f_filt_low, f_filt_high,tp_obsr, tp_theory , az_T):
    
    
    #rec=accel()

    #rec.read_v1(filname)
    L1=np.loadtxt(filname,usecols=[0])
    V1=np.loadtxt(filname,usecols=[1])
    T1=np.loadtxt(filname,usecols=[2])
    dt=0.005

    L=np.array([])
    V=np.array([])
    T=np.array([])

    # ############################# shifting the signal
   # if (tp_theory-tp_obsr)>=0:
   #     n_pad=int((tp_theory-tp_obsr)/dt)
   #     L=np.lib.pad(rec.L, (n_pad, 0), 'linear_ramp', end_values=(rec.L[0],0))
   #     V=np.lib.pad(rec.V, (n_pad, 0), 'linear_ramp', end_values=(rec.V[0],0))
   #     T=np.lib.pad(rec.T, (n_pad, 0), 'linear_ramp', end_values=(rec.T[0],0))
   # else:
   #     n_pad=-1*int((tp_theory-tp_obsr)/dt)
   #     L=rec.L[n_pad:]
   #     V=rec.V[n_pad:]
   #     T=rec.T[n_pad:]
    if (tp_theory-tp_obsr)>=0:
        n_pad=int((tp_theory-tp_obsr)/dt)
        L=np.lib.pad(L1, (n_pad, 0.), 'linear_ramp', end_values=(L1[0],0.))
        V=np.lib.pad(V1, (n_pad, 0.), 'linear_ramp', end_values=(V1[0],0.))
        T=np.lib.pad(T1, (n_pad, 0.), 'linear_ramp', end_values=(T1[0],0.))
    else:
        n_pad=-1*int((tp_theory-tp_obsr)/dt)
        L=L1[n_pad:]
        V=V1[n_pad:]
        T=T1[n_pad:]   
# ################ Rotation of signal
    teta = az_T*2 * np.pi / 360
    E =  T *  np.sin(teta) -  L* np.cos(teta)
    N =  T *  np.cos(teta) + L * np.sin(teta)


    # ############################# assign of signal to obspy stream

    tr1=Trace()#Trace:An object containing data of a continuous series, such as a seismic trace.
    tr2=Trace()#Each Trace object has the attribute 'data' pointing to a NumPy 'ndarray' of the actual time series and the attribute 'stats' which contains all meta information in a dict-like Stats object.
    tr3=Trace()
    sT=Stream(traces=[tr1, tr2, tr3]) # Streams are list-like objects which contain multiple Trace objects, i.e. gap-less continuous time series and related header/meta information.

    sT[0].stats.sampling_rate=200.0	 # Sampling rate in hertz (default value is 1.0)=1.0/0.005sec=200.0 Hz.
    sT[0].stats.delta=0.005	    	 # Sample distance in seconds (default value is 1.0).
    sT[0].data=E/100.               	 # convert to m/s/s
    sT[0].stats.npts=len(sT[0].data)     # Number of sample points (default value is 0, which implies that no data is present).
###    sT[0].stats.starttime=UTCDateTime(2009, 1, 1, 12, 0, 0) # Date and time of the first data sample given in UTC (default value is “1970-01-01T00:00:00.0Z”).

    sT[1].stats.sampling_rate=200.0
    sT[1].stats.delta=0.005
    sT[1].data=V/100.
    sT[1].stats.npts=len(sT[0].data)
###    sT[1].stats.starttime=UTCDateTime(2009, 1, 1, 12, 0, 0)

    sT[2].stats.sampling_rate=200.0
    sT[2].stats.delta=0.005
    sT[2].data=N/100.
    sT[2].stats.npts=len(sT[0].data)
###    sT[2].stats.starttime=UTCDateTime(2009, 1, 1, 12, 0, 0)

    sT.detrend()			# Remove a linear trend from all traces.	
    

    # ############################# prefiltering of the signal with linear phase filter with corner frequency equal to 0.4xdownsampling frequency
    if f_filt_low>0.4*sT[0].stats.sampling_rate/n_down:
        f_filter=0.4*sT[0].stats.sampling_rate/n_down
    else:
        f_filter=f_filt_low

    st_f=sT.copy()		# Return a deepcopy of the Stream object.The two objects are not the same But they have equal data (before applying further processing).
    st_f[0].data= obspy.signal.lowpass(sT[0].data, f_filter , corners=4, zerophase=True, df=sT[0].stats.sampling_rate)
    st_f[1].data= obspy.signal.lowpass(sT[1].data, f_filter , corners=4, zerophase=True, df=sT[1].stats.sampling_rate)
    st_f[2].data= obspy.signal.lowpass(sT[2].data, f_filter , corners=4, zerophase=True, df=sT[2].stats.sampling_rate)

#    if f_filt_high!=0.0:
#        st_f[0].data= obspy.signal.highpass(st_f[0].data, f_filt_high , corners=4, zerophase=True, df=st[0].stats.sampling_rate)
#        st_f[1].data= obspy.signal.highpass(st_f[1].data, f_filt_high , corners=4, zerophase=True, df=st[1].stats.sampling_rate)
#        st_f[2].data= obspy.signal.highpass(st_f[2].data, f_filt_high , corners=4, zerophase=True, df=st[2].stats.sampling_rate)


    st_d=st_f.copy()   
    st_d[0]=st_d[0].decimate(factor=n_down, no_filter=True) #Downsample data in all traces of stream by an integer factor.Deactivates automatic filtering if set to True. Defaults to False.
    st_d[1]=st_d[1].decimate(factor=n_down, no_filter=True)
    st_d[2]=st_d[2].decimate(factor=n_down, no_filter=True)


# ################## Integrating the signals to displacement
    dt=st_d[0].stats.delta
    
    EV=inte(st_d[0].data, dt)
    ED=inte(EV, dt)
    st_d[0].data=ED
  
    VV=inte(st_d[1].data, dt)
    VD=inte(VV, dt)
    st_d[1].data=VD
    
    NV=inte(st_d[2].data, dt)
    ND=inte(NV, dt)
    st_d[2].data=ND

    st_d1=st_d.copy()
    st_d2=st_d.copy()

    if f_filt_high!=0.0:
        st_d1[0].data= obspy.signal.highpass(st_d[0].data, f_filt_high , corners=4, zerophase=False, df=st_d[0].stats.sampling_rate)
        st_d1[1].data= obspy.signal.highpass(st_d[1].data, f_filt_high , corners=4, zerophase=False, df=st_d[1].stats.sampling_rate)
        st_d1[2].data= obspy.signal.highpass(st_d[2].data, f_filt_high , corners=4, zerophase=False, df=st_d[2].stats.sampling_rate)

        st_d2[0].data= obspy.signal.highpass(st_d[0].data, f_filt_high , corners=4, zerophase=True, df=st_d[0].stats.sampling_rate)
        st_d2[1].data= obspy.signal.highpass(st_d[1].data, f_filt_high , corners=4, zerophase=True, df=st_d[1].stats.sampling_rate)
        st_d2[2].data= obspy.signal.highpass(st_d[2].data, f_filt_high , corners=4, zerophase=True, df=st_d[2].stats.sampling_rate)

    return st_d1, st_d2, st_d, dt # st_d -> Displacement

    
# #################################################################################################################
# #################################################################################################################
# this function adjuct length of a signal, if the desired length be less than the lenght of the signal, it will be cropped
# and if the length of the signal be less than it, it will padded with zeros
# Inputs:   rec: np array of signal
#               n: number of desired length
#
# Output    rec_out: adjusted signal
#
# Written by Hadi Panahi (93.09.27)
# #################################################################

def adjust_length(rec, n):
    ll=len(rec)
    
    if n<=ll:
        rec_out=rec[0:ll]
    else:
        n_pad=n-ll
        rec_out=np.lib.pad(rec, (0, n_pad), 'constant', constant_values=(0,0))
        
    return rec_out
        
    
# #################################################################################################################
# #################################################################################################################
# Description:  Computes elastic response spectra for SDOF systems with Period
#               T and damping ratio xi subjected to a support acceleration f(t)
#               sampled at a step dt. Zero intial conditions are assumed.
#
#               ydd(t) + 2 xi w yd(t) + w^2  y(t) = -f(t)   y(0)=0, yd(0)=0
#
# Input:        f   = numpy array contanint Excitation vector
#               dt  = sample time in f
#               T   = numpy array containing periods of interest
#               xi  = numpy array containing damping rations (0.05 for 5%)
#
# Output:       SD  = Relative displacement response spectrum
#               PSV = Pseudo-relative-velocity
#               PSA = Pseudo-absolute-acceleration
#               SA  = Absolute acceleration
#               SV  = Relative velocity
#
# Written by Hadi Panahi (94.07.15)
# #################################################################

def sdof_response(f,dt,T,xi):
    
    y=np.zeros((2, len(f)))
    ind_xi=-1
    ind_T=-1
    SD=np.zeros((len(T), len(xi)))
    PSV=np.zeros((len(T), len(xi)))
    PSA=np.zeros((len(T), len(xi)))
    SV=np.zeros((len(T), len(xi)))
    SA=np.zeros((len(T), len(xi)))
    ED= np.zeros((len(T), len(xi)))
    
    for kis in xi:
        ind_xi+=1
        for per in T:
            ind_T+=1
            w=2*np.pi/per
            C=2*kis*w
            K=w*w
            y[:,0]=np.array([0, 0])
            A=np.array([[0, 1], [-K, -C]])
            Ae=expm(A*dt)
            
            AeB=np.dot(LA.solve(A, (Ae-np.eye(2))), np.array([[0], [-1]]))
            
            for k in range(1, len(f)):
                y[:,k]=np.dot(Ae, y[:,k-1])+(AeB*f[k])[:, 0]
            
            SD[ind_T,ind_xi]=np.max(abs(y[0,:]))
            PSV[ind_T,ind_xi]=SD[ind_T,ind_xi]*w
            PSA[ind_T,ind_xi]=SD[ind_T,ind_xi]*w*w
            SV[ind_T,ind_xi]=np.max(abs(y[1,:]))
            SA[ind_T,ind_xi]=np.max(abs(K*y[0,:]+C*y[1,:]))
            
            
    return SD, PSV, PSA, SV, SA


# #################################################################################################################
# #################################################################################################################
# this function calculates the SI value of a accelrogram
# Inputs:   f: np array of signal
#           dt: sampling period
#
# Output    SI: the SI value
#
# Written by Hadi Panahi (94.07.18)
# #################################################################

def SI_cal(f,dt):
    
    dT=0.1
    T=np.arange(0.1,2.6,dT)
    y=np.zeros((2, len(f)))
    ind_T=-1
    SV=np.zeros(len(T))
    kis=0.2
    
    for per in T:
        ind_T+=1
        w=2*np.pi/per
        C=2*kis*w
        K=w*w
        y[:,0]=np.array([0, 0])
        A=np.array([[0, 1], [-K, -C]])
        Ae=expm(A*dt)
         
        AeB=np.dot(LA.solve(A, (Ae-np.eye(2))), np.array([[0], [-1]]))
            
        for k in range(1, len(f)):
            y[:,k]=np.dot(Ae, y[:,k-1])+(AeB*f[k])[:, 0]
            
        SV[ind_T]=np.max(abs(y[1,:]))
        SI=(np.trapz(SV,dx=dT))/2.4
            
    return SI


# #################################################################################################################
# #################################################################################################################
# This function calculates the CAV value of a accelrogram according to EPRI 2005
# Inputs:   f: np array of signal (cm/s/s) (the unit is important)
#           dt: sampling period
#
# Output    CAV: the CAV value in unit of g-sec
#
# Written by Hadi Panahi (94.07.18)
# #################################################################

def CAV_cal(f,dt):

    div=int(np.floor(len(f)/(1/dt)))
    ll=int(np.floor(1/dt))

    CAV=0
    for i in range(div-1):
        sig=f[i*ll:(i+1)*ll]
        pga=np.max(np.abs(sig))
        if pga>=0.025*981:
            CAV+=np.trapz(np.abs(sig),dx=dt)

    return CAV/981.0


# #################################################################################################################
# #################################################################################################################
# Usage y=filter_butter(X,ord,flag,fc,type,dt)
# This function filters the signal with Butterworth filter, this code was tested with matlab and is OK
# Inputs:       ord: Order of Butterworth filter
#               flag: =1 one direction (non-linear phase)
#                     =2 two direction (linear-phase)
#               fc: corner frequency
#               type: ="highpass" for high-pass filtering
#                     ="lowpass" for low-pass filtering
#               dt: Sampling time (sec)
#
# Output:       y: The filtered signal
#
# By: Hadi Panahi (94.10.15)
# *************************************************************************
# *************************************************************************
        
def filter_butter(X,order,flag,fc,ttype,dt):
    nyq = 0.5 /dt 
    Wc = fc / nyq
    b, a = butter(order, Wc, btype=ttype)

    if flag==1:
        y=lfilter(b,a,X)
    elif flag==2:
        y=filtfilt(b,a,X)        
    return y

    
















