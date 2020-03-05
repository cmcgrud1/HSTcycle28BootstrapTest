import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy import interpolate
# Code to test the significance of 6 planets in determining correlation of cloud formation.
# We have 3 different correlations on metallicity, linear, quadratic, and sigmodal.
#However, fitting a linear function on all 3 and seeing how strong the p and r values are

x0, L, k, C = -.02, 1.1e5, 20, 100 #sigmoid function constants
sigmoid = lambda x: (L/(1.0+np.exp(-k*(x-x0)))) + C

A, B, C = 400400, .3, 100 # quadractic function constants
quadratic = lambda x: 400400*(.3+x)**2 +100

m, b = 199800.00000000003, 60040.0 # line constants
linear = lambda X: m*X + b

PerDif = lambda true, calc: (abs(calc-true)/float(true))*100.0 # to caclulate the percent difference for the one actual line

NoisyX = lambda mu, std:np.random.normal(mu, std, 1)[0] #Noise was made with normal distriubtion where mean was metallicity values and std was errorbars

def BootMet_linFit(Zerrors, function, Z=np.array([-0.2,-0.02,0.06,0.09,0.14,0.16]), iterations = 1000): #To calculate a linear fit of metallicity and cloud    
    #top coverage given certain gaussian error bars of metallicity and a linear, exponential, and sigmoidal true fit
    
    #Where metallicity of -.3 has 0 coverage and metallicity of .2 is completely clear
    x = np.linspace(-.3,.2,num = 200) 
    y = function(x)#Clear to cloudy in terms of cloud height pressure. 10^5 Pa clear and 100 Pa completely cloudy
    f = interpolate.interp1d(x, y) # to get functional form of data
    
    #To calculate the cloud coverage given a certain metallicity and given assumed linear trend
    Cloud_top = f(Z)
    cnt = 0
    Slopes, Intercepts, R_values = np.zeros(iterations), np.zeros(iterations), np.zeros(iterations)
    P_values, Std_errs = np.zeros(iterations), np.zeros(iterations)
    if function == linear: # only calculate percent difference if the correlation is actually linear
    	SlpPerDifs = np.zeros(iterations)
    Zs = np.zeros((len(Z),iterations))
    while cnt < iterations:
        NoisyZ = np.zeros(len(Z))
        for z in range(len(Z)):
            NoisyZ[z] = NoisyX(Z[z], Zerrors[z])
        Zs[:,cnt] = NoisyZ
        Slopes[cnt], Intercepts[cnt], R_values[cnt], P_values[cnt], Std_errs[cnt] = stats.linregress(NoisyZ,Cloud_top)
        if function == linear: 
        	SlpPerDifs[cnt] = PerDif(m, Slopes[cnt])
        cnt += 1
    if function == linear:
    	return Zs, Cloud_top, Slopes, Intercepts, R_values, P_values, Std_errs, SlpPerDifs, y
    else:
    	return Zs, Cloud_top, Slopes, Intercepts, R_values, P_values, Std_errs, y

if __name__ == "__main__":
	Num = 10000
	num = int(Num/2) #pick the central fit
	planet_count = 6
	Zz =np.array([-0.2,-0.02,0.06,0.09,0.14,0.16])
	z = np.random.choice(Zz, planet_count, replace = False)
	z = Zz
	print 'z:', z
	error_source = 'MIKE'
	Zerrors = 0.05*np.ones(planet_count) #expected errors
	# Zerrors = np.array([.09,.11,.03,.04,.19,.08]) #errors from literature 
	lin_Zs, cloudT, lin_Slopes, lin_Intercepts, lin_R_values, lin_P_values, lin_Std_errs, SlpPerDifs, lin_y = BootMet_linFit(Zerrors, linear, iterations = Num, Z=z)
	print '\033[1mFor linear correlation with', str(planet_count),'planets and', error_source,'errorbars\033[0m'
	print 'True slope:', m
	print 'Average slope:', np.mean(lin_Slopes)
	print 'Slope average percent error (%):', np.mean(SlpPerDifs)
	print 'Average p_value (%):', np.mean(lin_P_values)*100.0 #Want at least 10% threshold
	print 'Average r_value:', np.mean(lin_R_values)
	print '\n\n'

	quad_Zs, cloudT, quad_Slopes, quad_Intercepts, quad_R_values, quad_P_values, quad_Std_errs, quad_y = BootMet_linFit(Zerrors, quadratic, iterations = Num, Z=z)
	print '\033[1mFor quadratic correlation with', str(planet_count),'planets and',error_source,'errorbars\033[0m'
	print 'True slope:', m
	print 'Average slope:', np.mean(quad_Slopes)
	print 'Average p_value (%):', np.mean(quad_P_values)*100.0 #Want at least 10% threshold
	print 'Average r_value:', np.mean(quad_R_values)	
	print '\n\n'

	sig_Zs, cloudT, sig_Slopes, sig_Intercepts, sig_R_values, sig_P_values, sig_Std_errs, sig_y = BootMet_linFit(Zerrors, sigmoid, iterations = Num, Z=z)
	print '\033[1mFor sigmoidal correlation with', str(planet_count),'planets and',error_source,'errorbars\033[0m'
	print 'True slope:', m
	print 'Average slope:', np.mean(sig_Slopes)
	print 'Average p_value (%):', np.mean(sig_P_values)*100.0 #Want at least 10% threshold
	print 'Average r_value:', np.mean(sig_R_values)	
	print '\n\n'

	# To plot everything together
	plt.figure("Bootstraps")
	# plt.title('Linear fit to various ')
	x = np.linspace(-.3,.3,num = 200) 
	plt.subplot(131)
	plt.plot(x, lin_y, 'k')
	plt.plot(x, lin_Slopes[num]*x + lin_Intercepts[num], 'r--')
	plt.plot(lin_Zs[:,num],cloudT, 'D', markersize=6)
	plt.ylim([0,1.17e5])
	# plt.xlim([-.32,.22])
	plt.xticks(fontsize=17)
	plt.yticks(fontsize=17)
	plt.ylabel('Cloud deck Pressure (Pa)',fontweight='bold', fontsize = 25)
	plt.xlabel('Metallicity, [Fe/H]',fontweight='bold', fontsize = 25)
	plt.grid(True)

	plt.subplot(132)
	plt.plot(x, quad_y, 'k')
	plt.plot(x, quad_Slopes[num]*x + quad_Intercepts[num], 'r--')
	plt.plot(quad_Zs[:,num],cloudT, 'D', markersize=6)
	plt.ylim([0,1.17e5])
	# plt.xlim([-.32,.22])
	plt.xticks(fontsize=17)
	plt.yticks(fontsize=17)
	plt.xlabel('Metallicity, [Fe/H]',fontweight='bold', fontsize = 25)
	plt.grid(True)

	plt.subplot(133)
	plt.plot(x, sig_y, 'k', label = 'True fit')
	plt.plot(x, sig_Slopes[num]*x + sig_Intercepts[num], 'r--', label = '"observed" fit')
	plt.plot(sig_Zs[:,num],cloudT, 'D', markersize=6, label = '"observed" data')
	plt.ylim([0,1.17e5])
	# plt.xlim([-.32,.22])
	plt.xticks(fontsize=17)
	plt.yticks(fontsize=17)
	plt.xlabel('Metallicity, [Fe/H]',fontweight='bold', fontsize = 25)
	plt.grid(True)
	legend_properties = {'size': 15, 'weight':'bold'}
	plt.legend(prop=legend_properties)
	# plt.tight_layout()
	plt.show()
	plt.close()
