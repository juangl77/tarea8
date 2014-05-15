all: alpha_beta.png

alpha_beta.png: lotka_volterra_obs.py lotka_volterra_obs.dat
	python lotka_volterra_obs.py

lotka_volterra_obs.dat: 
	wget https://raw.githubusercontent.com/forero/ComputationalMethodsData/master/homework/hw_8/lotka_volterra_obs.dat

clean: 
	rm *~ *.png 