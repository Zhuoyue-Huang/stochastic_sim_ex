import numpy as np
import matplotlib.pyplot as plt


def chi_sq_density(x, nu):
    """Return the density of chi-squared distribution at x given parameter nu."""
    if nu<=2 or nu%2:
        raise NotImplemented
    else:
        return x**(nu/2-1) * np.exp(-x/2) / (2**(nu/2)*np.math.factorial(nu//2-1))

def scaling_exp_density(x, lam, M):
    """Return the density of exponential distribution with scale factor M at x given mean lam."""
    if 0<lam<1/2:
        pdf = lam * np.exp(-lam*x)
        return M * pdf
    else:
        raise NotImplemented

def optimal_M(nu):
    """Return optimal scaling factor M with exponential proposal for chi-squared samping with parameter nu."""
    return np.exp(1-nu/2) * (nu/2)**(nu/2) / np.math.factorial(nu//2-1)

def exp_sampling(lam):
    """Sample from exponential distribution by inverse transform."""
    u = np.random.uniform()
    return -np.log(1-u) / lam

nu = 4     # chi-squared parameter
lam = 1/nu # optimal parameter for exponential proposal
M = optimal_M(nu)

samples = [] # list to store samples
n = 100000
for i in range(n):
    # rejection sampling
    x = exp_sampling(lam)
    u = np.random.uniform()
    if u<=chi_sq_density(x, nu)/scaling_exp_density(x, lam, M):
        samples.append(x)

# density of chi-squared and proposal distribution
x_range = np.linspace(0, 25, 200)
p = chi_sq_density(x_range, nu)
q = scaling_exp_density(x_range, lam, M)

# plot the histogram of the samples and the density
plt.figure(figsize=(15, 5), dpi=500)
plt.subplot(1, 2, 1)
plt.plot(x_range, p, color='r', linewidth=2)
plt.plot(x_range, q)
plt.ylim(0, q[0])
plt.legend(['Chi-squared distribution', 'Proposal distribution (exponetial with scaling)'])
plt.title(f"Density functions for sampling and optimal proposal distribution")

plt.subplot(1, 2, 2)
plt.plot(x_range, p, color='r', linewidth=2)
plt.hist(samples, bins=150, density=True, color='k')
plt.ylim(0, q[0])
plt.legend(['Chi-squared distribution', 'Sampling histogram'])
plt.title(f"Rejection sampling procedures for n={n}")

# print difference of results of theoretical and practical acceptance rates
print("Sampling acceptance rate:", len(samples)/n, "\nTheoretical acceptance rate:", np.round(1/M, 5), "\nDifference between two acceptance rates above:", abs(np.round(len(samples)/n-1/M, 5)))


# Q2
def chi_reject_sampler(nu):
    """Return a single sample from chi-squared distribution with parameter nu."""
    lam = 1/nu
    M = optimal_M(nu)
    while True:
        x = exp_sampling(lam)
        u = np.random.uniform()
        if u<=chi_sq_density(x, nu)/scaling_exp_density(x, lam, M):
            return x

def mixture_density(x, w, nu):
    """Return the density of a mixed chi-squared distribution at x for parameter nu with weights w."""
    return w[0]*chi_sq_density(x, nu[0]) + w[1]*chi_sq_density(x, nu[1]) + w[2]*chi_sq_density(x, nu[2])

w = [0.2, 0.5, 0.3] # probability mass function
cdf = np.cumsum(w)  # compute CDF
nu = [4, 16, 40]

mix_samples = [] # list to store samples
n =  100000
for i in range(n):
    u = np.random.uniform()
    for j in range(len(cdf)):
        # find the first element of cdf that is greater than u
        # this is the index of the nu that we will sample
        if u < cdf[j]:
            mix_samples.append(chi_reject_sampler(nu[j]))
            # the first time if holds , we break out of the for loop
            break

# plot the histogram of the samples and the density
plt.figure(figsize=(10, 5), dpi=500)
xx = np.linspace(0, 80 , 1000)
plt.plot(xx , mixture_density(xx, w, nu), color='r', linewidth=2)
plt.hist(mix_samples, bins=150, density=True, color='black')
plt.legend(['mixture of chi-squared distribution', 'Sampling historgram'])
plt.title("Samping procedure for mixed chi-squared distribution")
