def betaconv(res,alpha1, beta1, alpha2, beta2):
    #Set support
    import numpy as np
    from scipy.stats import beta
    x=np.arange(0,2.001,res)#0:res:2;
    #Individual Beta pdfs
    f1 = beta.pdf(x,alpha1,beta1)
    f2 = beta.pdf(x,alpha2,beta2)
    #Compute convolution
    y = np.convolve(f1, f2)
    #Reduce to [0..2] support
    y = y[0:len(x)]
    #Normalize (so that all values sum to 1/res)
    y = y / (sum(y) * res)
    return y
def betasumcdf(x, alpha1, beta1, alpha2, beta2):
    import numpy as np
    #if not (np.ndim(x)==1 and np.shape(x)[0]==1):
    #    print 'only implemented for onedimensional input'
    # Compute the PDF first (since we want the entire pdf rather than just
    # one value from it, using betaconv is computationally more efficient
    # than using betasumpdf)
    res = 0.001;
    c = betaconv(res, alpha1, beta1, alpha2, beta2)
    #Sum the PDF up to point x
    x=[x]
    y = np.zeros(len(x))
    for i in range(0,len(x)):
        idx = round(x[i]/res)
        if idx < 1:
            y[i] = 0
        elif idx > len(c):
            y[i] = 1
        else:
            y[i] = np.trapz(c[0:idx]) * res;
    return y
def betaavgcdf(x, alpha1, beta1, alpha2, beta2):
    return betasumcdf(2*x, alpha1, beta1, alpha2, beta2)
def bacc_p(C):
    #Get alphas and betas
    A1 = C[0,0] + 1
    B1 = C[0,1] + 1
    A2 = C[1,1] + 1
    B2 = C[1,0] + 1
    #Compute area under pdf below 0.5
    b_p = betaavgcdf(0.5,A1,B1,A2,B2)
    return b_p[0]
def bacc_mean(C):
    A1 = C[0,0] + 1
    B1 = C[0,1] + 1
    A2 = C[1,1] + 1
    B2 = C[1,0] + 1
    res = 0.001;
    x = np.arange(0,2.001,res)
    c = betaconv(res, A1, B1, A2, B2)
    b = sum(x*c/2) * res;
    return b
import numpy as np
C=np.array([[50,10],[10,50]])
print bacc_mean(C),bacc_p(C)