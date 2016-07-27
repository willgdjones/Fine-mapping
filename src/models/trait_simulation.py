### Sample genotypes

def simulate_genotype(n,m,geno_dist):
    """
    Simulate a genotype of n samples and m causal SNPs with specified genotype distribution for (0,1,2).
    """
    X=np.zeros([n,m])
    for i in range(m):
        X[:,i] = [np.random.choice(a=[0,1,2],p=geno_dist) for x in range(n)]
    return np.array(X)

###example
# X = simulate_genotype(n=10000,m=30,geno_dist=[0.85,0.1,0.05])

def simulate_traits(X,snp_group,eps=0.5):
    """
    SNPs in the form e.g. {3: 0.9, 5:0.4, 8:0.5}. Dictionary values are the linear model coefficients (beta values).
    eps is the level of unexplained variance. X is the genotype information.
    """
    beta = np.array(snp_group.values()).T
    snps = snp_group.keys()
    eps_vector = np.array(np.random.normal(0,eps,X.shape[0])).T
    return np.add(np.dot(X[:,snps], beta), eps_vector)

# examples
# y = simulate_traits(X,eps=0.5,snp_group={3: 5, 9: 3})

def build_linear_models(X,y):
    """
    Build univariate linear models for each SNP column in X against the trait y.
    """
    return [stats.linregress(X[:,i],y) for i in range(X.shape[1])]

# example
# models1 = [x for x in build_linear_models(X,y)]

def calc_effect_sizes(models):
    """
    Calculate the effect sizes = beta / se(beta) of individual SNPs towards the traits.
    Takes in a list of linear regression models.
    """
    return [x.slope / x.stderr for x in models]

# example
# z1 = [x.slope / x.stderr for x in models1]


