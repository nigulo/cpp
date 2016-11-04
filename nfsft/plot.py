
import matplotlib.pyplot as plt
import numpy as np

dat = np.loadtxt("data.txt", usecols=(0,1,2))
extent=[dat[:,0].min(),dat[:,0].max(),dat[:,1].min(),dat[:,1].max()]
plot_aspect=(extent[1]-extent[0])/(extent[3]-extent[2])*2/3
num_long = np.shape(np.unique(dat[:,0]))[0]
num_lat = np.shape(np.unique(dat[:,1]))[0]
dat=dat.reshape(num_long,num_lat,3)

reconst = np.loadtxt("reconst.txt", usecols=(0,1,2))
num_long = np.shape(np.unique(reconst[:,0]))[0]
num_lat = np.shape(np.unique(reconst[:,1]))[0]
reconst=reconst.reshape(num_long,num_lat,3)

cmin1=min(np.concatenate(dat[:,:,2]))
cmax1=max(np.concatenate(dat[:,:,2]))

cmin2=min(np.concatenate(reconst[:,:,2]))
cmax2=max(np.concatenate(reconst[:,:,2]))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(dat[:,:,2].T, cmap='gnuplot', extent=extent, vmin=cmin1, vmax=cmax1)
ax1.set(xlabel='longitude', ylabel='latitude');
ax1.set_aspect(aspect=plot_aspect)

ax2.imshow(reconst[:,:,2].T, cmap='gnuplot', extent=extent, vmin=cmin2, vmax=cmax2)
ax2.set(xlabel='longitude', ylabel='latitude');
ax2.set_aspect(aspect=plot_aspect)

plt.savefig("data.png")
plt.close()

spec = np.loadtxt("decomp.txt")
max_l = spec[:,0].max()
power=np.zeros(max_l+1)
for l,m,re,im in spec:
    #if (re > 1e-14) or (im > 1e-14):
    power[l] += re**2+im**2 / (2*l+1)
power /= (num_long*num_lat)
ls = np.arange(0,np.shape(power)[0])
indices = np.where(power > 1e-24)
power=power[indices]
ls=ls[indices]

plt.plot(ls, power, 'b-', )
plt.yscale('log')
#plt.autoscale(enable=True, axis='y')
#plt.xlim((3, 51))
plt.savefig("power.png")
plt.close()
