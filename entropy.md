记录一些常用公式和启发式的观点用于帮助记忆。
## 熵
当我们观测到了一个随机变量x的真实值，怎么度量我们获得了多少信息（降低了多少未知性/混乱）？

if p(x) > p(y) then h(x) < h(y) 
h(x,y) = h(x) + h(y) 以及 p(x, y) = p(x)p(y)
联想到log(xy) = log(x) + log(y)
那么h(x) = -log(p(x))
熵： H[x] = E[h(x)] 
离散： 
![](https://images-1256319930.cos.ap-shanghai.myqcloud.com/concreteEntropy.gif)

连续： 
![](https://images-1256319930.cos.ap-shanghai.myqcloud.com/CodeCogsEqn.gif)

H[x]只和p(x)的形式有关，和具体的值无关，是x这个随机变量的信息量的平均度量

### 另一个角度(物理学)理解熵
假设把N个物体放到一堆桶里，设第i个桶放ni个，那么分配物体的方法共有：

W = ![](https://images-1256319930.cos.ap-shanghai.myqcloud.com/bucketNObj.gif)

熵（多样性）的定义：

![](https://images-1256319930.cos.ap-shanghai.myqcloud.com/entropy_physics.gif)

使用斯特林公式：

![](https://images-1256319930.cos.ap-shanghai.myqcloud.com/stirling_eq.gif)

得到：

![](https://images-1256319930.cos.ap-shanghai.myqcloud.com/entropy_appro.gif)

如果我们把这些桶看作一个随机变量的概率p(X = xi)=pi, 那么就和之前的定义一致了。 
### 熵的大小
总体原则：分布越均匀，熵越大

![](https://images-1256319930.cos.ap-shanghai.myqcloud.com/entropy_values.PNG) 

### 条件熵
给定x的情况下，获得y的值所需的信息量的期望：

![](https://images-1256319930.cos.ap-shanghai.myqcloud.com/conditional_entro.png) 

另外，推导可得：

![](https://images-1256319930.cos.ap-shanghai.myqcloud.com/union_entro.png) 

### 相对熵（KL散度）和互信息
**KL散度**： 描述两个分布p(x)和q(x)之间的差异，是用q来作为p编码时，传输x的值所需的平均信息的度量。
定义：

![](https://images-1256319930.cos.ap-shanghai.myqcloud.com/kl_div.png)

从定义可看出KL(pq) !≡ KL(qp).
可由KL的凸性和琴生不等式推导出： KL(pq) >= 0 
只有当p(x)=q(x)处处相等时，KL=0成立

**互信息**： 描述x和y的相关度, 通过p(x,y)和p(x)p(y)的KL散度来衡量
定义：

![](https://images-1256319930.cos.ap-shanghai.myqcloud.com/mutual_info.png)

可见，只有当x,y相互独立，即p(x,y) = p(x)p(y)时，I[x,y] = 0
另外，推导可得：
I[x, y] = H[x] − H[x|y] = H[y] − H[y|x].

### 韦恩图

![](https://images-1256319930.cos.ap-shanghai.myqcloud.com/entropy_weyen.jpg)
