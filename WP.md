# 面向高速公路人机共享驾驶的双向信任权限决策与自适应鲁棒控制方法

## 摘要

本文面向高速公路人机共享驾驶场景，研究驾驶人意图、机器意图、双向信任、共享权限决策与车辆控制之间的协同建模问题。现有方法通常将驾驶人意图预测、机器决策、权限分配和轨迹控制作为相对独立的模块处理，难以在驾驶人意图存在误差、机器决策存在效率偏好、周围交通环境动态变化的情况下同时保证安全性、效率性、舒适性和人类意图一致性。为此，本文提出一种“预测-信任-权限-控制”一体化的人机共享驾驶框架：首先构建风险感知的驾驶人决策与控制意图联合预测模型；然后基于规则和势场构建可解释的机器决策与控制意图生成模型；进一步利用双向信任评估和区间二型 TSK 模糊推理生成参考权限序列；在此基础上使用 Transformer 强化学习模型对参考权限进行增量优化，得到真实执行权限；最后通过共享意图驱动的自适应鲁棒 MPC-lite 控制器实现车辆闭环执行。

关键词：人机共享驾驶；驾驶意图预测；双向信任；共享权限；强化学习；鲁棒控制；highD

## 1. 研究问题定义

本文研究的问题不是 highD 数据集本身的缺失或噪声问题，而是高速公路人机共享驾驶中的权限协同与安全控制问题。highD 数据集仅作为高速公路交通场景、驾驶人轨迹和验证环境的实验支撑。

给定时刻 `k` 的道路、交通、自车、驾驶人历史状态和预测风险信息：

```math
X_k = \{X_{\mathrm{road}}, X_{\mathrm{veh}}, X_{\mathrm{ego}}, X_{\mathrm{drv}}, R_k^p\}
```

本文目标是学习或构造一个共享驾驶决策控制函数：

```math
\Pi:
X_k \mapsto
\left\{
c_k^{share},
\lambda_{h,k},
\delta_k,
a_k,
\tau_{k:k+N}
\right\}
```

其中，`c_k^{share} \in \{L,S,R\}` 表示最终共享决策，分别对应左换道、直行和右换道；`\lambda_{h,k}` 表示驾驶人真实执行权限，`1-\lambda_{h,k}` 表示机器执行权限；`\delta_k` 和 `a_k` 分别为前轮转角和纵向加速度；`\tau_{k:k+N}` 表示未来预测窗口内的执行轨迹。

本文需要解决以下核心科学问题：

1. 如何同时预测驾驶人的高层决策意图和连续控制意图，并评估其未来执行风险。
2. 如何构造不依赖学习模型、可解释且安全优先的机器决策与控制意图。
3. 如何从“机器对人”和“人对机器”两个方向刻画信任，并将双向信任映射为参考共享权限。
4. 如何在参考权限基础上进一步优化真实执行权限，使系统能在高风险和人机冲突场景中主动修正不安全融合结果。
5. 如何将人机融合意图转化为稳定、舒适、安全且高效的车辆控制输入。

## 2. 总体技术路线

整体框架由五个模块构成：

```text
Work1: 驾驶人决策与控制意图预测
Work2: 机器决策与控制意图生成
Work3: 双向信任评估与参考权限推理
Work4: RL-based 真实权限优化与自适应鲁棒 MPC-lite 控制执行
```

系统信息流为：

```text
交通环境 X_k
  -> 驾驶人意图模型: (P_h, c_h, delta_h, v_h, a_h, tau_h)
  -> 机器意图模型:   (P_m, c_m, delta_m, v_m, a_m, tau_m)
  -> 双向信任评估:   (T_{m->h}, T_{h->m})
  -> 模糊参考权限:   lambda_h^{ref}
  -> RL 权限优化:    lambda_h^{RL}
  -> 人机意图融合:   (c_share, delta_share, v_share, a_share)
  -> 自适应鲁棒 MPC-lite 控制:  (delta_cmd, a_cmd, tau_exec)
```

最终执行控制不是直接采用驾驶人预测结果，也不是直接采用机器规划结果，而是通过真实权限 `lambda_h^{RL}` 对人机意图进行融合：

```math
u_k^{share}
=
\lambda_{h,k}^{RL} u_k^h
+
(1-\lambda_{h,k}^{RL})u_k^m
```

其中：

```math
u_k = [\delta_k, v_k, a_k]
```

共享决策概率为：

```math
P_{share}(c|X_k)
=
\lambda_{h,k}^{RL}P_h(c|X_k)
+
(1-\lambda_{h,k}^{RL})P_m(c|X_k)
```

最终共享决策为：

```math
c_k^{share}
=
\arg\max_{c\in\{L,S,R\}}P_{share}(c|X_k)
```

## 3. 创新点一：风险感知的驾驶人决策与控制意图联合预测模型

### 3.1 研究动机

驾驶人意图不仅包含“左换道、右换道、直行”等离散决策，还包含未来转角、速度、加速度等连续控制意图。仅预测离散换道标签无法支撑后续共享权限和控制执行。因此，本文构建驾驶人高层决策和连续控制意图的多任务联合预测模型。

### 3.2 模型输入

驾驶人模型输入包括：

```math
X_k^h
=
\{X_{\mathrm{road}}, X_{\mathrm{front}}, X_{\mathrm{rear}},
X_{\mathrm{left}}, X_{\mathrm{right}},
X_{\mathrm{ego}}^{hist}, X_{\mathrm{drv}}^{hist}, R_k^p\}
```

其中，预测风险势场由周围车辆距离、相对速度和车道关系构造：

```math
R_{\mathrm{veh}}
=
\exp\left(-\frac{d^2}{\sigma_d^2}\right)
(1+\alpha|\Delta v|)
```

多车风险场为：

```math
R_k^p
=
\sum_{i=1}^{M}
\omega_i R_{\mathrm{veh}}^i
```

### 3.3 模型结构

历史状态编码器：

```math
H_k = f_h(X_{\mathrm{ego}}^{hist},X_{\mathrm{drv}}^{hist},X_{\mathrm{veh}})
```

风险势场编码器：

```math
G_k = f_r(R_k^p)
```

特征融合：

```math
F_k = \mathrm{Fusion}(H_k,G_k)
```

多任务输出头：

```math
[P_h(c_k|X_k), \hat{\delta}_{k:k+N}^h,
\hat{v}_{k:k+N}^h,
\hat{a}_{k:k+N}^h,
\hat{t}_{event}^h]
=
f_{\theta_h}(F_k)
```

其中：

```math
P_h(c_k|X_k)
=
\mathrm{Softmax}(W_cF_k+b_c)
```

### 3.4 损失函数

联合损失为：

```math
L_h
=
L_{\mathrm{cls}}
+\lambda_{\delta}L_{\delta}
+\lambda_vL_v
+\lambda_aL_a
+\lambda_{event}L_{event}
+\lambda_{smooth}L_{smooth}
```

离散决策损失：

```math
L_{\mathrm{cls}}
=
-\sum_{c\in\{L,S,R\}}y_c\log P_h(c|X_k)
```

连续控制意图损失：

```math
L_{\delta}
=
\frac{1}{N}\sum_{j=1}^{N}
\left|\hat{\delta}_{k+j}^h-\delta_{k+j}\right|
```

```math
L_v
=
\frac{1}{N}\sum_{j=1}^{N}
\left|\hat{v}_{k+j}^h-v_{k+j}\right|
```

```math
L_a
=
\frac{1}{N}\sum_{j=1}^{N}
\left|\hat{a}_{k+j}^h-a_{k+j}\right|
```

事件时间损失：

```math
L_{event}
=
\left|\hat{t}_{event}^h-t_{event}\right|
```

平滑约束：

```math
L_{smooth}
=
\sum_{j=2}^{N}
\left(
|\hat{\delta}_{k+j}^h-\hat{\delta}_{k+j-1}^h|
+
|\hat{a}_{k+j}^h-\hat{a}_{k+j-1}^h|
\right)
```

该模块输出驾驶人的预测决策、预测控制意图和预测轨迹，为后续信任评估、权限推理和控制融合提供人类意图输入。

## 4. Work2：规则与势场驱动的机器决策与控制意图生成模型

### 4.1 研究动机

机器意图模块用于表达自动驾驶系统在当前环境下的安全优先决策。该模块不采用学习方法，而采用规则、人工势场和可解释安全约束实现，使其具备稳定性和可解释性。

### 4.2 机器风险势场

对候选决策 `c \in \{L,S,R\}`，构造候选轨迹：

```math
\tau_m^c = \mathcal{G}(c, X_k)
```

候选轨迹风险为：

```math
\rho_m^c
=
w_d \phi_d^c
+w_{ttc}\phi_{ttc}^c
+w_{rear}\phi_{rear}^c
+w_{lane}\phi_{lane}^c
+w_{col}\phi_{col}^c
```

其中：

```math
\phi_d^c = \exp(-d_{\min}^c/d_0)
```

```math
\phi_{ttc}^c = \exp(-TTC^c/\tau_0)
```

```math
\phi_{col}^c =
\begin{cases}
1, & \text{candidate trajectory has collision risk}\\
0, & \text{otherwise}
\end{cases}
```

### 4.3 效率收益

机器同时考虑交通效率：

```math
E_m^c
=
w_v(v_{target}-v_{ego})
+w_{front}\max(0,v_{ego}-v_{front})
+w_{gap}g_c
```

其中，`g_c` 表示目标车道可用间隙。

### 4.4 机器决策

机器候选决策代价定义为：

```math
J_m(c)
=
\alpha_{\rho}\rho_m^c
-\alpha_E E_m^c
+\alpha_{comfort}J_{comfort}^c
+\alpha_{rule}J_{rule}^c
```

机器最终决策为：

```math
c_m
=
\arg\min_{c\in\{L,S,R\}}J_m(c)
```

机器连续控制意图由候选轨迹反解得到：

```math
[\hat{\delta}_{k:k+N}^m,\hat{v}_{k:k+N}^m,\hat{a}_{k:k+N}^m]
=
\mathcal{K}^{-1}(\tau_m^{c_m})
```

其中，`\mathcal{K}^{-1}` 表示逆运动学映射。

## 5. 创新点二：双向信任驱动的参考权限生成模型

### 5.1 研究动机

人机共享驾驶中，权限不应固定分配，也不应仅由驾驶人意图或机器意图单独决定。本文提出双向信任建模：机器需要评估是否信任驾驶人，人也需要根据机器意图与自身意图的一致性和收益评估是否信任机器。

### 5.2 机器对人的信任

将驾驶人预测控制意图输入运动学模型：

```math
\tau_h
=
\mathcal{M}
(\hat{\delta}_{k:k+N}^h,\hat{v}_{k:k+N}^h,\hat{a}_{k:k+N}^h)
```

在交通环境中计算该轨迹的执行风险：

```math
\rho_h
=
w_{ttc}\phi_{ttc}(\tau_h)
+w_d\phi_d(\tau_h)
+w_{col}\phi_{col}(\tau_h)
+w_{lat}\phi_{lat}(\tau_h)
```

机器对人的信任为：

```math
T_{m\rightarrow h}
=
\exp(-\gamma_h\rho_h)
```

即驾驶人预测轨迹风险越高，机器对人的信任越低。

### 5.3 人对机器的信任

人对机器的信任由人机决策差异、控制意图差异、机器风险和机器效率收益共同决定：

```math
T_{h\rightarrow m}
=
\sigma
\left(
\eta_0
-\eta_cD_c
-\eta_uD_u
-\eta_r\rho_m
+\eta_eE_m
\right)
```

决策差异：

```math
D_c
=
1-\sum_{c\in\{L,S,R\}}P_h(c|X_k)P_m(c|X_k)
```

控制意图差异使用预测窗口内所有时刻，并对近时刻赋予更大权重：

```math
D_u
=
\sum_{j=1}^{N}\omega_j
\left(
\alpha_{\delta}
|\hat{\delta}_{k+j}^{h}-\hat{\delta}_{k+j}^{m}|
+
\alpha_v
|\hat{v}_{k+j}^{h}-\hat{v}_{k+j}^{m}|
+
\alpha_a
|\hat{a}_{k+j}^{h}-\hat{a}_{k+j}^{m}|
\right)
```

时间权重为：

```math
\omega_j
=
\frac{\exp(-\beta j)}
{\sum_{l=1}^{N}\exp(-\beta l)}
```

### 5.4 区间二型 TSK 模糊参考权限推理

参考权限由双向信任、风险和人机意图差异共同决定：

```math
\lambda_h^{ref}
=
\mathcal{F}_{IT2-TSK}
(T_{m\rightarrow h},T_{h\rightarrow m},\rho,D_c,D_u)
```

第 `r` 条模糊规则为：

```math
\mathcal{R}_r:
\text{If } z_1 \text{ is } \tilde{A}_{r1},...,
z_Q \text{ is } \tilde{A}_{rQ},
\text{ then }
\lambda_{h,r}^{ref}
=
a_{r0}+\sum_{q=1}^{Q}a_{rq}z_q
```

其中，`\tilde{A}_{rq}` 为区间二型模糊集合。规则激活强度为：

```math
\underline{\gamma}_r
=
\prod_{q=1}^{Q}
\underline{\mu}_{\tilde{A}_{rq}}(z_q)
```

```math
\overline{\gamma}_r
=
\prod_{q=1}^{Q}
\overline{\mu}_{\tilde{A}_{rq}}(z_q)
```

参考权限输出为：

```math
\lambda_h^{ref}
=
\frac{
\sum_{r=1}^{M}\bar{\gamma}_r w_r \lambda_{h,r}^{ref}
}{
\sum_{r=1}^{M}\bar{\gamma}_r w_r
}
```

规则参数通过离线优化获得：

```math
\Theta_f^*
=
\arg\min_{\Theta_f}J_f(\Theta_f)
```

优化目标：

```math
J_f
=
J_{safety}
+\beta_1J_{human}
+\beta_2J_{smooth}
+\beta_3J_{eff}
```

该模块输出的是参考权限，而不是最终执行权限。

## 6. 创新点三：基于 Transformer-RL 的真实共享权限优化模型

### 6.1 研究动机

模糊推理得到的参考权限具有可解释性，但在复杂交互场景中可能无法完全满足安全和效率目标。本文进一步构建强化学习权限优化器，在参考权限基础上学习增量修正，使系统在高风险和人机冲突场景中获得更可靠的真实执行权限。

真实权限定义为：

```math
\lambda_h^{RL}
=
\mathrm{clip}(\lambda_h^{ref}+\Delta\lambda,0,1)
```

其中：

```math
\Delta\lambda
=
\pi_{\theta}^{RL}(o_k)
```

### 6.2 观测空间

RL 观测包括参考权限、双向信任、风险、人机意图差异和关键交通交互特征：

```math
o_k
=
[
\lambda_h^{ref},
T_{m\rightarrow h},
T_{h\rightarrow m},
\rho_h,
\rho_m,
D_c,
D_u,
d_{front},
\Delta v_{front},
TTC,
g_{left},
g_{right}
]
```

为了利用时序信息，采用 Transformer Encoder 编码历史观测：

```math
Z_k
=
\mathrm{TransformerEncoder}(o_{k-H:k})
```

策略网络：

```math
\Delta\lambda_k
=
\pi_{\theta}(Z_k)
```

价值网络：

```math
Q_{\phi}(Z_k,\Delta\lambda_k)
```

### 6.3 强化学习奖励

奖励函数设计为：

```math
r_k
=
r_{safety}
+r_{eff}
+r_{comfort}
+r_{human}
+r_{smooth}
```

安全奖励：

```math
r_{safety}
=
w_1\mathbb{I}_{no\ collision}
-w_2\max(0,d_{safe}-d_{min})
-w_3\phi_{ttc}
```

效率奖励：

```math
r_{eff}
=
-w_4|v_{target}-v_{ego}|
```

舒适性奖励：

```math
r_{comfort}
=
-w_5|a|
-w_6|\dot{a}|
-w_7|\dot{\delta}|
```

人类一致性奖励：

```math
r_{human}
=
-w_8\lambda_h^{RL}\rho_h
-w_9(1-\lambda_h^{RL})D_u
```

权限平滑奖励：

```math
r_{smooth}
=
-w_{10}|\lambda_{h,k}^{RL}-\lambda_{h,k-1}^{RL}|
```

训练时构造 hard case buffer，重点采样参考权限失败、人机冲突、邻车风险高、前车慢车阻挡和并行风险场景，使 RL 主要学习关键场景下的权限修正。

## 7. 创新点四：共享意图驱动的自适应鲁棒 MPC-lite 控制器

### 7.1 研究动机

共享权限优化后得到的是融合意图，而车辆仍需要可执行、稳定且安全的控制输入。本文设计自适应鲁棒 MPC-lite 控制器，将融合意图生成的参考轨迹作为跟踪目标，同时考虑预测误差、车辆稳定性、安全距离、舒适性和效率。

### 7.2 参考轨迹生成

由共享控制意图生成参考轨迹：

```math
x_{k+1}^{ref}
=
x_k^{ref}
+v_k^{share}\cos\psi_k^{ref}\Delta t
```

```math
y_{k+1}^{ref}
=
y_k^{ref}
+v_k^{share}\sin\psi_k^{ref}\Delta t
```

```math
\psi_{k+1}^{ref}
=
\psi_k^{ref}
+\frac{v_k^{share}}{L}\tan(\delta_k^{share})\Delta t
```

该参考轨迹是人机意图融合后的期望轨迹，MPC-lite 的作用是稳定跟踪和安全修正，而不是重新进行高层决策。

### 7.3 车辆动力学模型

控制器采用简化动力学自行车模型：

```math
\dot{x}=v_x\cos\psi-v_y\sin\psi
```

```math
\dot{y}=v_x\sin\psi+v_y\cos\psi
```

```math
\dot{\psi}=r
```

```math
\dot{v}_x=a_x+v_yr
```

```math
\dot{v}_y=\frac{F_{yf}+F_{yr}}{m}-v_xr
```

```math
\dot{r}=\frac{l_fF_{yf}-l_rF_{yr}}{I_z}
```

轮胎侧向力：

```math
F_{yf}=C_f\alpha_f,\quad F_{yr}=C_r\alpha_r
```

并满足附着约束：

```math
|F_y|\leq \mu F_z
```

### 7.4 MPC-lite 优化目标

控制输入：

```math
u_k=[\delta_k,a_k]
```

优化目标：

```math
u_k^*
=
\arg\min_{u_k}J_{mpc}
```

```math
J_{mpc}
=
w_{tr}J_{track}
+w_{safe}J_{safety}
+w_{st}J_{stable}
+w_{com}J_{comfort}
+w_{eff}J_{eff}
+w_{rob}J_{robust}
```

轨迹跟踪项：

```math
J_{track}
=
\|p_k-p_k^{ref}\|^2
+\alpha_{\psi}\|\psi_k-\psi_k^{ref}\|^2
+\alpha_v\|v_k-v_k^{ref}\|^2
```

安全项：

```math
J_{safety}
=
\max(0,d_{safe}-d_{min})^2
+\alpha_{col}\mathbb{I}_{collision}
```

稳定性项：

```math
J_{stable}
=
\left(\frac{\beta}{\beta_{max}}\right)^2
+\alpha_r\left(\frac{r}{r_{max}}\right)^2
```

舒适性项：

```math
J_{comfort}
=
\alpha_a a^2
+\alpha_{\dot{a}}\dot{a}^2
+\alpha_{\dot{\delta}}\dot{\delta}^2
```

效率项：

```math
J_{eff}
=
|v_{target}-v_{ego}|^2
```

鲁棒项用于抑制驾驶人预测误差导致的控制突变：

```math
J_{robust}
=
\|u_k-u_k^{share}\|^2
+\alpha_e\|\tau_k-\tau_k^{share}\|^2
```

### 7.5 自适应权重

多目标权重根据稳定裕度、风险程度、信任程度和权限大小动态调整：

```math
w_{safe}(k)
=
w_{safe}^0
\left[1+\alpha_{\rho}\rho(k)+\alpha_T(1-T_{m\rightarrow h}(k))\right]
```

```math
w_{st}(k)
=
w_{st}^0
\left[1+\alpha_s(1-S_{stab}(k))\right]
```

```math
w_{tr}(k)
=
w_{tr}^0
\left[1+\alpha_{\lambda}\lambda_h^{RL}(k)T_{h\rightarrow m}(k)\right]
```

稳定裕度定义为：

```math
S_{stab}
=
1-\max
\left(
\frac{|\beta|}{\beta_{max}},
\frac{|r|}{r_{max}}
\right)
```

当环境风险升高或机器对人的信任降低时，控制器提高安全权重；当车辆稳定裕度降低时，提高稳定性权重；当人类权限较高且人对机器信任较高时，提高对共享参考轨迹的跟踪权重。

## 8. 实验验证设计

### 8.1 数据与场景来源

本文使用 highD 作为高速公路交通场景和驾驶人行为数据来源。由于目标验证场景需要覆盖典型人机冲突情况，实验采用“highD 原型 + 可控环境车微调”的方式构造验证场景：

```text
highD 原始道路和车辆轨迹
  -> 选取相似交通原型
  -> 微调周围环境车位置、速度、加速度
  -> 注入闭环交通环境
  -> 验证 Work1-Work4 联合系统
```

需要强调的是，环境车微调后的状态会输入算法和碰撞检测，而不是只用于可视化显示。

### 8.2 典型验证场景

构造 7 类典型验证场景：

```text
case1: 驾驶人换道意图不合理，目标侧风险高，系统应直行减速。
case2: 驾驶人左换道插空风险高，机器认为右换道更安全，系统应右换道。
case3: 驾驶人左换道方向合理但轨迹激进，系统保持左换道方向并平滑轨迹。
case4: 前车慢且距离近，直行效率低，系统应安全换道提效。
case5: 驾驶人直行意图合理但速度过快，邻道阻塞，系统应直行减速。
case6: 主车与旁车并行且旁车加速，机器超车代价过大，系统应遵从人类减速。
case7: 主车与旁车并行但旁车速度不足，人机均可加速超越，系统应加速通过。
```

### 8.3 评价指标

安全性：

```math
I_{col}=0,\quad d_{min}>d_{safe},\quad TTC>TTC_{min}
```

效率性：

```math
E = -|v_{target}-v_{ego}|
```

舒适性：

```math
C = -(|a|+|\dot{a}|+|\dot{\delta}|)
```

人类一致性：

```math
H = -D_u
```

稳定性：

```math
S_{stab}>0
```

综合评价：

```math
J_{eval}
=
\kappa_1 I_{safe}
+\kappa_2 E
+\kappa_3 C
+\kappa_4 H
+\kappa_5 S_{stab}
```

## 9. highD 数据处理在本文中的作用

highD 数据处理属于实验实现层，不构成本文研究问题本身。其作用包括：

1. 提供真实高速公路车辆轨迹，用于训练驾驶人意图预测模型。
2. 提供道路、车道、前后车和邻车信息，用于构建机器决策、信任评估和验证环境。
3. 通过逆运动学从轨迹估计前轮转角，用于连续控制意图监督。
4. 对车辆类型进行筛选或统一尺寸处理，使控制验证中的 ego/control 车辆为 passenger car。
5. 通过 highD 原型和环境车微调构造覆盖人机冲突、安全风险、效率收益和并行风险的验证场景。

## 10. 预期学术贡献

本文预期贡献如下：

1. 提出风险感知的驾驶人决策与控制意图联合预测模型，实现离散决策、连续控制和事件时间的统一预测。
2. 构建规则与势场驱动的机器意图生成方法，使机器决策兼具安全性、效率性和可解释性。
3. 提出基于执行风险和意图差异的人机双向信任评估方法，并通过区间二型 TSK 模糊推理生成参考共享权限。
4. 提出基于 Transformer-RL 的真实权限增量优化方法，在参考权限基础上针对高风险 hard cases 进行安全修正。
5. 设计共享意图驱动的自适应鲁棒 MPC-lite 控制器，实现预测误差下的安全、稳定、舒适和高效控制。
6. 构建 highD 注入式闭环验证环境，通过典型人机冲突场景验证所提方法的有效性。
