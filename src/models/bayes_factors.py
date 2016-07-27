### Create selection of SNPs
def select_snps(z, subset):
    return [z[i] for i in subset]

#example
# for subset in it.combinations(range(len(z1)),3):
#     print subset, select_snps(z1, subset)



### Select covariance submatrix

def select_cov(cov, subset):
    return cov[np.ix_(subset,subset)]

#example
#select_cov(LD_tss_1, (0,1,5))

### Calculate Bayes Factor

def calc_BF(z, cov,n,v=0.1):
    """
    Calculate the Bayes factor of a single set of candidate SNPs effect sizes z,
    covariance matrix cov, a prior variance on beta v, and a sample
    size n.
    """
    z = np.matrix(z)
    z = z.T
    v_matrix = np.matrix(np.eye(len(z)) * v)
#     pdb.set_trace()
    coeff = 1. / math.sqrt(np.linalg.det((np.matrix(np.eye(len(z))) + n * v_matrix * np.matrix(cov))))
    exponent = 0.5* z.T * np.matrix(np.linalg.pinv((n*v_matrix).I + cov)) * z
    return np.array(math.log(coeff) + exponent)[0][0]

# example
# subset = (0,1,5,8)
# cov = select_cov(LD_tss_1, subset)
# z = select_snps(z1, subset)
# v = np.eye(len(z))/1000
# n = 1000
# calc_BF(z,cov,v,n)

def calc_prior(x,m,prior='binomial'):
    if prior == 'binomial':
        p = 1./m
        l = len(x)
        return p**l * (1-p)**(m-l)
    else:
        return None

# example
# calc_prior((1,3,5),30)

def calc_posterior(variant_set_BF,prior='binomial'):

    priors = [math.log(calc_prior(x[0],30)) for x in variant_set_BF]

    log_bayes_factors = [x[1] for x in variant_set_BF]

    unscaled_log_posteriors = [ log_bayes_factors[i] + priors[i] for i in range(len(log_bayes_factors))]

    scaled_log_posteriors = np.array(unscaled_log_posteriors) - max(unscaled_log_posteriors)

    scaled_posteriors = [math.exp(x) for x in scaled_log_posteriors]

    calib_factor = sum([math.exp(x) for x in scaled_log_posteriors])

    posteriors = [x/calib_factor for x in [math.exp(x) for x in scaled_log_posteriors]]

    aug_posteriors = [(variant_set_BF[i][0], posteriors[i]) for i in range(len(posteriors))]

    aug_posteriors.sort(key=lambda x: x[1], reverse=True)

    return aug_posteriors





def calc_variant_set_BFs(data,k,v=0.1,prior='binomial'):
    """
    Calculate variant set posteriors with a binomial prior as normal,
    searching all variant sets up till size k.
    v is the prior variance on beta.
    data has the format (z,LD,n) where z is the effect sizes,
    LD is the linkage disequilibrium matrix, and n is the
    number of samples.
    """
    bayes_factors = []
    for i in range(1,k):
        for subset in it.combinations(range(len(data[0])),i):
            z = select_snps(data[0], subset)
            cov = select_cov(data[1],subset)
            n = data[2]
            bayes_factors.append((subset, calc_BF(z, cov,n,v)))

    bayes_factors.sort(key=lambda x: x[1], reverse=True)
    return bayes_factors


