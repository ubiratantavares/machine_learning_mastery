
**What is the most enjoyable thing when you read a new paper? For me, this is the following:**

Imagine a popular model suddenly getting upgraded — with just a few elegant tweaks.

Three years after _DeepAR_ [1], Amazon engineers published its revamped version, known as _**Deep GPVAR**_ **[2]** _**(D**eep **G**aussian-**P**rocess **V**ector **A**uto-**r**egressive)_ or simply _GPVAR_. Some papers also call it _GP-Copula._

This is a much-improved model of the original version. Plus, it’s open-source! In this article, we discuss:

- **How** _**Deep GPVAR**_ **works in depth.**
    
- **How DeepAR and** _**Deep GPVAR**_ **are different.**
    
- **What problems does** _**Deep GPVAR**_ **solve and why it’s better than DeepAR?**
    
- **A hands-on tutorial on energy demand forecasting.  
    **
    

> _Find the hands-on project for Deep GPVAR in the [AI Projects folder](https://aihorizonforecast.substack.com/p/ai-projects), along with other cool projects!_

### _**  
What is Deep GPVAR?**_

Deep GPVAR is an autoregressive DL model that leverages low-rank Gaussian Processes to model thousands of time-series jointly, by considering their interdependencies.

Let’s briefly review the advantages of using _Deep GPVAR :_

- **Multiple time-series support:** The model uses multiple time-series data to learn global characteristics, improving its ability to forecast accurately.
    
- **Extra covariates:** _Deep GPVAR_ allows extra features (covariates).
    
- **Scalability:** The model leverages _low-rank gaussian distribution_ to scale training to multiple time series **simultaneously.**
    
- **Multi-dimensional modeling:** Compared to other global forecasting models, _Deep GPVAR_ models time series together, rather than individually. This allows for improved forecasting by considering their interdependencies.
    

The last part is what differentiates _Deep GPVAR from DeepAR_. We will discuss this more in the next section.

## **  
Global Forecasting Mechanics**

A global model trained on multiple time series is not a new concept. But why the need for a global model?

At my previous company, where clients were interested in time-series forecasting projects, the main request was something like this:

> _“We have 10,000 time-series, and we would like to create a single model, instead of 10,000 individual models.”_

The time series could represent product sales, option prices, atmospheric pollution, etc. — it doesn't matter. What’s important here is that a company needs a lot of resources to train, evaluate, deploy, and monitor (for **concept drift**) _10,000_ time series in production.

So, that is a good reason. Also, at that time, there was no _N-BEATS_ or [Temporal Fusion Transformer](https://aihorizonforecast.substack.com/p/ai-projects).

However, if we are to create a global model, what should it learn? Should the model just learn a clever mapping that conditions each time series based on the input? But, this assumes that time series are **independent**.

**Or should the model learn global temporal patterns that apply to all time series in the dataset?**

### **  
Interdependencies Of Time Series**

_Deep GPVAR_ builds upon _DeepAR_ by seeking a more advanced way to utilize the dependencies between multiple time series for improved forecasting.

For many tasks, this makes sense.

A model that considers time series of a global dataset as independent loses the ability to effectively utilize their relationships in applications such as finance and retail. For instance, risk-minimizing portfolios require a forecast of the covariance of assets, and a probabilistic forecast for different sellers must consider competition and cannibalization effects.

**Therefore, a robust global forecasting model cannot assume the underlying time series are independent.**

_Deep GPVAR is differentiated from DeepAR_ in two things:

- **High-dimensional estimation:** _Deep GPVAR_ **jointly** models time series together, factoring in their relationships. For this purpose, the model estimates their **covariance matrix** using a **low-rank Gaussian approximation**.
    
- **Scaling:** _Deep GPVAR_ does not simply normalize each time series, like its predecessor. Instead, the model learns how to scale each time series by transforming them first using **Gaussian Copulas.**
    

The following sections describe how these two concepts work in detail.

### **  
Low-Rank Gaussian Approximation — Introduction**

As we said earlier, one of the best ways to study the relationships of multiple time series is to estimate the covariate matrix.

However, scaling this task for thousands of time series is not easily accomplished — due to memory and numerical stability limitations. Plus, covariance matrix estimation is a time-consuming process — the covariance should be estimated for every time window during training.

To address this issue, the authors simplify the covariance estimation using **low-rank approximation**.

Let’s start with the basics. Below is the matrix form of a Multivariate normal`N∼(μ,Σ)` with mean `μ ∈ (k,1)` and covariance `Σ ∈ (k,k)`

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F1da7a114-694a-4a6d-bc27-f9b1bc6c3462_412x67.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F1da7a114-694a-4a6d-bc27-f9b1bc6c3462_412x67.png)

**Equation 1:** Multivariate Gaussian distribution in matrix form

The problem here is the size of the covariance matrix `Σ` that is quadratic with respect to `N`, the number of time series in the dataset.

We can address this challenge using an approximated form, called the **low-rank Gaussian approximation.** This method has its roots in factor analysis and is closely related to SVD (Singular Value Decomposition).

Instead of computing the full covariance matrix of size `(N,N)`**,** we can approximate by computing instead:

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F2b265acf-c0bf-48f6-a181-8b6607be36a6_211x41.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F2b265acf-c0bf-48f6-a181-8b6607be36a6_211x41.png)

**Equation 2:** Low-rank covariance matrix formula

where `D ∈ R(N,N)` is a diagonal matrix and `V ∈ R(N,r)`.

But why do we represent the covariance matrix using the low-rank format? Because since`r<<N`, it is proved that the Gaussian likelihood can be computed using `O(Nr² + r³)` operations instead of `O(N³)` (the proof can be found in the paper’s Appendix).

The low-rank normal distribution is part of PyTorch’s **distributions** module. Feel free to experiment and see how it works:

```
# multivariate low-rank normal of mean=[0,0], cov_diag=[1,1], cov_factor=[[1],[0]]

# covariance_matrix = cov_diag + cov_factor @ cov_factor.T

m = LowRankMultivariateNormal(torch.zeros(2), torch.tensor([[1.], [0.]]), 
torch.ones(2))
m.sample()  
#tensor([-0.2102, -0.5429])
```

### **  
Deep GPVAR Architecture**

> _**Notation:** From now on, all variables in bold are considered either vectors or matrices._

Now that we have seen how low-rank normal approximation works, we delve deeper into _Deep GPVAR’s_ architecture.

First, _Deep GPVAR_ is similar to _DeepAR —_ the model also uses an LSTM network. Let’s assume our dataset contains `N` time series, indexed from `i= [1…N]`

At each time step `t` we have:

1. An LSTM cell takes as input the target variable `z_t-1` of the previous time step `t-1` **for a subset of time series**. Also, the LSTM receives the hidden state `h_t-1` of the previous time step.
    
2. The model uses the LSTM to compute its hidden vector `h_t`.
    
3. The hidden vector `h_t` will now be used to compute the`μ, Σ` parameters of a multivariate Gaussian distribution `N∼(μ,Σ)`. This is a special kind of normal distribution called **Gaussian copula (**More about that later**).**
    

This process is shown in **Figure 1:**

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F45d6bf13-f968-4a58-9f9d-92174cbf810a_692x360.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F45d6bf13-f968-4a58-9f9d-92174cbf810a_692x360.png)

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F5dcdb5cc-ffbf-466a-8d10-c1f6b87763cf_692x360.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F5dcdb5cc-ffbf-466a-8d10-c1f6b87763cf_692x360.png)

**Figure 1:** Two training steps of Deep GPVAR. The covariance matrix Σ is expressed as Σ=D+V*V^T ([Source](https://arxiv.org/abs/1910.03002))

At each time step `t`, _Deep GPVAR_ randomly chooses `B << N` time-series to implement the low-rank parameterization: On the left, the model chooses from (1,2 and 4) time series and on the right, the model chooses from (1,3 and 4).

The authors in their experiments have configured `B = 20`. With a dataset containing potentially over `N = 1000` time series, the benefit of this approach becomes clear.

There are 3 parameters that our model estimates:

- The **means** `μ` of the normal distribution.
    
- The covariance matrix parameters are `d` and `v` according to **Equation 2.**
    

They are all calculated from the `h_t` LSTM hidden vector. **Figure 2** shows the low-rank covariance parameters:

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fde893f54-0886-4884-8678-335205d9f37c_700x88.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fde893f54-0886-4884-8678-335205d9f37c_700x88.png)

**Figure 2:** Parameterization of the low-rank covariance matrix, as expressed in **Equation 2**

> _Careful with notation:_ `μ_i`_,_ `d_i`_,_ `v_i` _refer to the_`i-th` _time series in our dataset, where_ `i ∈ [1..N]`_._

For each time-series `i`, we create the `y_i` vector, which concatenates `h_i`with `e_i` — the `e_i` vector contains the features/covariates of the `i-th` time series. Hence we have:

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F333f0a59-e220-4995-ac3e-cdb5ad2b402f_362x159.jpeg)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F333f0a59-e220-4995-ac3e-cdb5ad2b402f_362x159.jpeg)

**Figure 3** displays a training snapshot for a time step `t`:

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F48f7c8c7-27f8-4694-b39f-e2bff73bf8c3_700x541.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F48f7c8c7-27f8-4694-b39f-e2bff73bf8c3_700x541.png)

**Figure 3:** A detailed architecture of low-rank parameterization

Notice that:

- `μ` and `d` are scalars, while `v` is vector. For example, `μ = w_μ^T * y` with dimensions `(1,p)`*`(p,1`), therefore `μ` is a scalar.
    
- The same LSTM is used for all time series, including the dense layer projections `w_μ` ,`w_d` ,`w_u`.
    

Hence, the neural network parameters `w_μ` ,`w_d` ,`w_u`and`y` are used to compute the `μ` and `Σ` parameters which are shown in **Figure 3.**

### **  
The Gaussian Copula Function**

> _Question: What does the_ `μ` _and_ `Σ` _parameterize?_

They parameterize a special kind of multivariate Gaussian distribution, called **Gaussian Copula.**

> _But why does Deep GPVAR needs a Gaussian Copula?_

Remember, _Deep GPVAR_ does joint multi-dimensional forecasting, so we cannot use a simple univariate Gaussian, like in _DeepAR_.

> _Ok. So why not use our familiar multivariate Gaussian distribution instead of a Copula function — like the one shown in **Equation 1?**_

2 reasons:

**1)** **Because a multivariate Gaussian distribution requires gaussian random variables as marginals.** We could also use mixtures, but they are too complex and not applicable in every situation. Conversely, Gaussian Copulas are easier to use and can work with any input distribution — and by input distribution, we mean an individual time series from our dataset.

Hence, the copula learns to estimate the underlying data distributions without making assumptions about the data.

**2) The Gaussian Copula can model the dependency structure among these different distributions by controlling the parameterization of the covariance matrix** `Σ`_**.**_ That’s how _Deep GPVAR_ learns to consider the interdependencies among input time series, something other global forecasting models don’t do.

> _**Remember: time series can have unpredictable interdependencies.** For example, in retail, we have product cannibalization: a successful product pulls demand away from similar items in its category. So, we should also factor in this phenomenon when we forecast product sales. With copulas, Deep GPVAR learns those interdependencies automatically._

### **  
What are Copulas?**

A copula is a mathematical function that describes the correlation between multiple random variables.

Copulas are heavily used in quantitative finance for portfolio risk assessment. [Their misuse also played a significant role in the 2008 recession.](http://samueldwatts.com/wp-content/uploads/2016/08/Watts-Gaussian-Copula_Financial_Crisis.pdf)

More formally, a copula function `C` is a CDF of `N` random variables, where each random variable (marginal) is uniformly distributed:

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F8816ddc1-9f8b-496c-90af-783fe4a7e163_571x77.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F8816ddc1-9f8b-496c-90af-783fe4a7e163_571x77.png)

**Figure 4** below shows the plot of a bivariate copula, consisting of 2 marginal distributions. The copula is defined in the [0–1]² domain (x, y-axis) and outputs values in [0–1] (z-axis):

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fd389d9a1-092c-4aa0-ae6c-d2d8da6f8fc3_700x548.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fd389d9a1-092c-4aa0-ae6c-d2d8da6f8fc3_700x548.png)

**Figure 4:** A Gaussian copula CDF function consisting of 2 beta distributions as marginals

A popular choice for `C` is the Gaussian Copula — the copula of **Figure 4** is also Gaussian.

### **  
How we construct Copulas**

We won’t delve into much detail here, but let’s give a brief overview.

Initially, we have a random vector — a collection of random variables. In our case, each random variable `z_i` represents the observation of a time series `i` at a time step `t`:

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F00885051-a127-415b-9187-ff5de421da29_226x53.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F00885051-a127-415b-9187-ff5de421da29_226x53.png)

Then, we make our variables **uniformly distributed** using the _probability integral transform:_ The CDF output of any continuous random variable is uniformly distributed,`F(z)=U` :

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fadd4f235-09d4-4eb8-b2a5-cadcba1c246e_422x46.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fadd4f235-09d4-4eb8-b2a5-cadcba1c246e_422x46.png)

And finally, we apply our Gaussian Copula:

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc3eaaa8a-6855-46ae-b0de-4777d37ecfe6_620x53.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc3eaaa8a-6855-46ae-b0de-4777d37ecfe6_620x53.png)

where **Φ**^-1 is the inverse standard gaussian CDF `N∼(0,1)`, and **φ** (lowercase letter) is a gaussian distribution parameterized with `μ` and `Σ` _._ Note that `Φ^-1[F(z)] = x`, where `x~(-Inf, Inf)` because we use the standard inverse CDF.

**So, what happens here?**

We can take any continuous random variable, marginalize it as uniform, and then transform it into a gaussian. The chain of operations is the following:

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fb9c4c0c5-f9ef-468e-bc1b-30581b649b4e_187x53.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fb9c4c0c5-f9ef-468e-bc1b-30581b649b4e_187x53.png)

There are 2 transformations here:

- `F(z) = U`, known as _probability integral transform_. Simply put, this transformation converts any continuous random variable to uniform.
    
- `Φ^-1(U)=x`, known as _[inverse sampling](https://en.wikipedia.org/wiki/Inverse_transform_sampling)_. Simply put, this transformation converts any uniform random variable to the distribution of our choice — here `Φ` is gaussian, so `x` also becomes a gaussian random variable.
    

In our case, `z` are the past observations of a time series in our dataset. Because our model makes no assumptions about how the past observations are distributed, we use the **empirical CDF —** a special function that calculates the CDF of any distribution non-parametrically (empirically).

In other words, at the `F(z) = U` transformation, `F` is the empirical CDF and not the actual gaussian CDF of the variable `z` . The authors use `m=100` past observations throughout their experiments to calculate the empirical CDF.

### **  
Recap of Copulas**

> _To sum up, the Gaussian copula is a multivariate function that uses_ `μ` _and_ `Σ` _to directly paremeterize the correlation of two or more random variables._

But how does a Gaussian copula differ from a Gaussian multivariate probability distribution(PDF)? Besides, a Gaussian Copula is just a multivariate CDF.

1. The Gaussian Copula can use any random variable as a marginal, not just a Gaussian.
    
2. The original distribution of the data does not matter — using the probability integral transform and the empirical CDF, we can transform the original data to gaussian, **no matter how they are distributed**.
    

### **  
Gaussian Copulas in Deep GPVAR**

Now that we have seen how copulas work, it’s time to see how _Deep GPVAR_ uses them.

Let’s go back to **Figure 3.** Using the LSTM, we have computed the `μ` and `Σ` parameters. What we do is the following:

**Step 1: Transform our observations to Gaussian**

Using the copula function, we transform our observed time-series datapoints `z` to gaussian `x` using the copula function. The transformation is expressed as:

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F8ae9ae5f-798f-4f88-a673-1a5e42c44cc5_501x45.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F8ae9ae5f-798f-4f88-a673-1a5e42c44cc5_501x45.png)

where `f(z_i,t)` is actually the marginal transformation `Φ^-1(F(z_i))` of the time-series `i`.

In practice, our model makes no assumptions about how our past observations `z` are distributed. Therefore, no matter how the original observations are distributed, our model can effectively learn their behavior, including their correlation, thanks to Gaussian Copulas' power.

**Step 2: Use the computed parameters for the Gaussian.**

I mentioned that we should transform our observations to Gaussian, but what are the parameters of the Gaussian? In other words, when I said that `f(z_i) = Φ^-1(F(z_i))`, what are the parameters of `Φ` ?

The answer is the `μ` and `Σ` parameters — these are calculated from the dense layers and the LSTM shown in **Figure 3.**

**Step 3: Calculate the loss and update our network parameters**

To recap, we transformed our observations to Gaussian and we assume those observations are parameterized by a low-rank normal Gaussian. Thus, we have:

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F300d3737-fbf4-43f8-af59-64daa98a96eb_700x60.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F300d3737-fbf4-43f8-af59-64daa98a96eb_700x60.png)

where `f1(z1)` is the transformed observed prediction for the first time series, `f2(z2)` refers to the second one and `f_n(z_n)` refers to the N-th time series of our dataset.

Finally, we train our model by maximizing the **multivariate** **gaussian log-likelihood function.** The paper uses the convention of minimizing the loss function **—** the gaussian log-likelihood preceded with a minus:

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F301b2ee9-0c60-4d53-a860-f625509817fb_242x122.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F301b2ee9-0c60-4d53-a860-f625509817fb_242x122.png)

Using the gaussian log-likelihood loss, _Deep GPVAR_ updates its LSTM and the shared dense layer weights displayed in **Figure 3.**

Also, notice:

- The `z` is not a single observation, but the vector of observations from all `N` time-series at time `t`. The summation loops until `T`, the maximum lookup window — upon which the gaussian log-likelihood is evaluated.
    
- And since **z** is a vector of observations, the gaussian log-likelihood is actually **multivariate**. In contrast, DeepAR uses a univariate gaussian likelihood.
    

### **  
Deep GPVAR Variants**

From the current paper, Amazon created 2 models, **Deep GPVAR** (which we describe in this article) and **DeepVAR**.

DeepVAR is similar to _Deep GPVAR._ The difference is that DeepVAR uses a global multivariate LSTM that receives and predicts all time series at once. On the other hand, _Deep GPVAR_ unrolls the LSTM on each time series separately.

In their experiments, the authors refer to the DeepVAR as **Vec-LSTM** and _Deep GPVAR as_ **GP**_._

- The _Vec-LSTM_ and _GP_ terms are mentioned in **Table 1** of the original paper.
    
- The _Deep GPVAR_ and _DeepVAR_ terms are mentioned in Amazon’s Forecasting library Gluon TS.
    

This article describes the _**Deep GPVAR**_ **variant**, which is better on average and has fewer parameters than _DeepVAR._ Feel free to read the original paper and learn more about the experimental process.