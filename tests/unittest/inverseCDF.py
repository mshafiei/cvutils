from scipy.stats import norm
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1)
mean, var, skew, kurt = norm.stats(moments='mvsk')
x = np.linspace(norm.ppf(0.01),
                norm.ppf(0.99), 100)
ax.plot(x, norm.pdf(x),
       'r-', lw=5, alpha=0.6, label='norm pdf')

pdf = norm.pdf(x)
# u = np.random.rand(10)
a = 100
b = 50
n = 4
# u = np.array([1/n] * n)
u = np.linspace(0,1,100)
sampled = sampleInvCDF1D(pdf/pdf.sum(),u)
ax.scatter(x[sampled],[1] * len(sampled))
plt.show()