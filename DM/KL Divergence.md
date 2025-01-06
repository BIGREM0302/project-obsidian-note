
### KL Divergence
$$\text{As a method to measure the difference between P:target and Q:mode, or the price when compressing information}$$

$$KL(P||Q) = \sum_{x\in \chi}P(x)log\frac{P(x)}{Q(x)}:discrete$$
$$KL(P||Q) = \int\limits_{\chi}P(x)log\frac{P(x)}{Q(x)}:continuous$$
- Properties
	- Non negativity:
		-$KL(P||Q)\geq0$ 
		-$KL(P||P)=0$
	- Anti symmetric:
		-$KL(P||Q)\neq KL(Q||P)$

