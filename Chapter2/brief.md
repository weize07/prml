​	本章用贝叶斯+共轭先验分布的方式解决伯努利分布、multinomial分布和高斯分布的参数估计问题。

​	这种方法和极大后验估计 (MAP) 的区别在于，MAP的后验分布中仍然含有未知参数$ \bf w $， 需要进一步通过一个最优化问题来解决。而共轭先验分布通过观察到新的数据点，并使用贝叶斯公式将数据点的“知识”融入先验从而得到修正的后验分布。



#### 二项分布和Beta分布

伯努利分布：$ Bern(x|μ) = μ^x(1 − μ)^{1−x}; x \in \{0, 1\} $

$ D = \{x_1,...,x_N\},\ p(D|u) = \prod\limits_{n=1}^{N} u^{x_n}(1-u)^{1-x_n} $

二项分布：$ Bin(m|N, u) = \binom{N}{m}u^{m}(1-u)^{N-m} $

$ u $的先验——Beta分布:  $ Beta(u|a, b) = \frac{\Gamma (a+b)}{\Gamma(a)\Gamma(b)} u^{a-1}(1-u)^{b-1} $

观察了(m+l)个新的数据点D，其中m个1，l个0，那么后验：

$ p(u|m, l, a, b) \propto p(m,l|u)p(u|a,b) \propto u^m(1-u)^{l}u^{a-1}(1-u)^{b-1} = u^{m+a-1}(1-u)^{l+b-1} \propto Beta(u|m+a, l+b) $

所以$ p(x=1|D) = \frac{m+a}{m+l+a+b} $



