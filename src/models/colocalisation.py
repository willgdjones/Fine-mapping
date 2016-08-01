def is_colocalised(X, trait1, trait2,db=0):
    """
    With respect to a shared genotype X, Determine whether trait1 and trait2 are colocalised.
    I.e. whether there is evidence that they share a genetic basis.
    """

    ### generate individual linear models
    models1 = build_linear_models(X,y1)
    models2 = build_linear_models(X,y2)

    ### pull out slope and standard error terms.
    beta1 = [x.slope for x in models1]
    se_beta1 = [x.stderr for x in models1]

    beta2 = [x.slope for x in models2]
    se_beta2 = [x.stderr for x in models2]

    ### calculate z scores
    simulated_effectsize_data1 = ([x*np.sqrt(n) for x in beta1], LD_matrix, n)
    simulated_effectsize_data2 = ([x*np.sqrt(n) for x in beta2], LD_matrix, n)

    ### generate the gene set Bayes Factors
    gene_set_BFs1 = calc_variant_set_BFs(simulated_effectsize_data1,k=4,v=0.01)
    gene_set_BFs2 = calc_variant_set_BFs(simulated_effectsize_data2,k=4,v=0.01)


    ### calculate the posteriors
    gene_set_posteriors1 = calc_posterior(gene_set_BFs1)
    gene_set_posteriors2 = calc_posterior(gene_set_BFs2)

    if db == 1:

        print gene_set_BFs1[0:10]
        print gene_set_BFs2[0:10]

        print gene_set_posteriors1[0:10]
        print gene_set_posteriors2[0:10]


    ### sort by posterior size
    gene_set_posteriors1.sort(key=lambda x: x[0], reverse=False)
    gene_set_posteriors2.sort(key=lambda x: x[0], reverse=False)

    ### select just toe posteriors
    posteriors1 = [x[1] for x in gene_set_posteriors1]
    posteriors2 = [x[1] for x in gene_set_posteriors2]

    ### generate cartesian product from the posteriors
    cart_product = list(itertools.product(posteriors1,posteriors2))

    gene_set_len1 = len(gene_set_posteriors1)
    gene_set_len2 = len(gene_set_posteriors2)

    ### calculate colocalisation posteriors with a specificed scoring function.
    colocalisations = np.array(map(lambda x: min(x[0],x[1]), cart_product)).reshape(gene_set_len1,gene_set_len2)


    ### pull out sorted set list
    sorted_setlist1 = [x[0] for x in gene_set_posteriors1]
    sorted_setlist2 = [x[0] for x in gene_set_posteriors2]

    if db == 1:

        ###  create bidirectional map from gene_set to positon in colocalisation array
        setlist_1map = bidict([(sorted_setlist1[i],i) for i in range(len(sorted_setlist1))])
        setlist_2map = bidict([(sorted_setlist1[i],i) for i in range(len(sorted_setlist2))])

        bf_1map = dict(gene_set_BFs1)
        bf_2map = dict(gene_set_BFs2)


        posterior1_map = dict(gene_set_posteriors1)
        posterior2_map = dict(gene_set_posteriors2)
        pdb.set_trace()

    ### output total evidence for colocalisation
    return sum([colocalisations[i][i] for i in range(colocalisations.shape[0])])
