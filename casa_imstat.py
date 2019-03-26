import numpy as np

dir = "/Users/rcooke/Work/Research/Accel/data/TW_Hya/2016.1.00440.S/science_goal.uid___A001_X889_X18e/group.uid___A001_X889_X18f/member.uid___A001_X889_X190/product/"
fname = "TW_Hya_NO-tclean.image.pbcor.copy.fits"

my_stats = imstat(dir+fname, axes=[2,3])
#print(my_stats)
print(my_stats["rms"].shape)




