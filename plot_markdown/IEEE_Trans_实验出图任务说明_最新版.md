# IEEE Transactions 实验出图任务说明（最新版）

## 1. 任务目标

请你作为 **IEEE Transactions 论文实验出图专家和代码执行助手**，基于我的人机共享决策与鲁棒控制论文逻辑，重新设计并生成一套完整、清晰、可用于论文主文的实验结果图表。

本任务不是简单美化图片，而是要让每一张图明确服务于论文结论。每张图必须回答以下问题之一：

1. 人机意图冲突在哪里？
2. 驾驶人意图、机器意图和共享意图之间有什么差异？
3. 本文方法如何通过双向信任和 RL 权限优化生成更合理的共享行为参考？
4. 本文 adaptive robust MPC 执行后是否能够稳定、平滑、安全地跟踪共享参考？
5. 本文方法相比基准方法是否提升了安全性、稳定性、意图一致性和驾驶人操作舒适性？

最终需要输出：

1. 完整的图表设计方案；
2. 每类图对应的具体图片清单；
3. 每张图的横轴、纵轴、变量、图例、线型建议；
4. 每张图的保存路径和文件名；
5. 每张图支撑的论文结论；
6. 每张图的英文图题；
7. MATLAB 或 Python 绘图代码；
8. 所有图片单独保存，不要合并成大图。

---

## 2. 论文实验逻辑

我的论文逻辑如下：

人机共驾系统中，驾驶人与自动化系统可能在战术决策层存在意图冲突，例如换道方向、机动时机、速度策略和轨迹偏好不同。本文首先对驾驶人意图和机器意图进行解耦建模，然后基于双向信任关系生成参考权限，再通过强化学习方法优化未来权限序列，最后由 adaptive robust MPC 执行共享控制。

实验图表需要体现以下逻辑链条：

1. **人机意图冲突确实存在**；
2. **本文方法能够识别并协调人机意图差异**；
3. **双向信任变化能够影响参考权限分配**；
4. **RL 优化后的权限序列比参考权限更加平滑、合理**；
5. **adaptive robust MPC 能够在权限变化和权限扰动下保持车辆稳定性和跟踪性能**；
6. **本文方法能够提升安全性、意图一致性、车辆稳定性，并降低驾驶人操作负荷**。

---

## 3. 总体出图要求

### 3.1 IEEE Transactions 风格要求

所有图形风格应参考 **IEEE Transactions** 论文标准：

1. 图形简洁、清晰、专业；
2. 避免花哨背景、渐变、阴影和复杂装饰；
3. 优先采用黑白打印友好的线型、标记和颜色组合；
4. 颜色不宜过多，应保证灰度打印下仍可区分；
5. 字体建议使用 Times New Roman；
6. 坐标轴标签、图例、刻度字体大小应适合 IEEE 双栏排版；
7. 线宽、标记大小应保证缩放后仍清晰可读；
8. 所有图应有明确的横纵坐标、单位和图例；
9. 所有图应保证单独阅读时仍能理解其含义；
10. 图内标题尽量简洁，正式图题放在论文 caption 中。

### 3.2 图例强制要求

图例是当前出图的主要问题之一，必须严格处理：

1. 图例不得遮挡道路、车辆、轨迹、冲突区域、关键时间点、稳定边界、阈值线和峰值位置；
2. 轨迹图的图例优先放在图外上方、图外下方或图外右侧；
3. 如果图内有足够空白区域，图例可以放在图内空白区，但不得遮挡任何有效数据；
4. 对于轨迹图，不允许图例覆盖换道区域、障碍车附近区域和人机意图冲突区域；
5. 对于时序图，不允许图例覆盖峰值、突变点、阈值线和关键时间窗口；
6. 对于相平面图，不允许图例覆盖稳定边界和主要相轨迹；
7. 必要时可使用无边框图例、透明背景图例或单独导出图例；
8. 如果图例过多，必须拆图，而不是把大量图例强行塞进一张图；
9. 每张图生成后必须自检：图例是否遮挡核心数据。如果遮挡，必须自动调整或重新绘制。

### 3.3 图像清晰度要求

1. 每张图优先保存为矢量图格式，例如 `.pdf`、`.eps` 或 `.svg`；
2. 同时保存一份 `.png` 用于快速预览；
3. PNG 分辨率建议不低于 600 dpi；
4. 线条不宜过细；
5. 图中文字不宜过小；
6. 坐标范围应合理，避免曲线挤压或留白过多；
7. 不同曲线之间应有清晰区分；
8. 关键区域可以适当使用局部放大图，但不要造成图面拥挤；
9. 图例、文字标注、箭头说明不得遮挡关键实验结果。

### 3.4 单图输出要求

注意：**每一类图不是一张图，而是一类图组。**

1. 每一类图可以包含多个 Case、多个指标、多个对比方法；
2. 每一类图应放在一个独立子文件夹中；
3. 每个子图必须单独保存；
4. 不要把多个子图汇总成一个大图；
5. 不要输出 `(a)(b)(c)(d)` 形式的大型拼接图；
6. 每个 Case、每个指标、每个对比结果尽量单独输出为一张清晰图片；
7. 后续如果论文排版需要，再由作者自行决定是否组合为多子图。

错误示例：

```text
Fig. 6(a)-(d) combined figure
```

正确示例：

```text
Case1_Trajectory_HM_Conflict.pdf
Case1_Authority_Profile.pdf
Case1_PhasePlane_Stability.pdf
Case1_LateralAcceleration.pdf
```

---

## 4. 文件夹组织要求

请按照以下方式组织输出文件：

```text
Experiment_Figures/
│
├── Fig_Type_0_Scenario_Overview/
│   ├── Case1_Scenario_Illustration.pdf
│   ├── Case2_Scenario_Illustration.pdf
│   └── Scenario_Overview_Table.xlsx
│
├── Fig_Type_1_Trajectory/
│   ├── Case1_HM_Intention_Conflict.pdf
│   ├── Case1_Shared_Reference_Generation.pdf
│   ├── Case1_Method_Execution_Comparison.pdf
│   ├── Case2_HM_Intention_Consistency.pdf
│   └── Case3_Risk_Avoidance_Trajectory.pdf
│
├── Fig_Type_2_Authority/
│   ├── Case1_Authority_Reference_vs_RL.pdf
│   ├── Case1_Bidirectional_Trust.pdf
│   ├── Case2_Authority_Reference_vs_RL.pdf
│   └── Case3_Authority_Profile.pdf
│
├── Fig_Type_3_Lateral_Dynamics/
│   ├── Case1_PhasePlane_Stability.pdf
│   ├── Case1_LateralAcceleration.pdf
│   ├── Case2_PhasePlane_Stability.pdf
│   └── Case2_LateralAcceleration.pdf
│
├── Fig_Type_4_Driver_Operation/
│   ├── Case2_DriverSteering_Time.pdf
│   ├── Case2_DriverSteering_Speed.pdf
│   └── Case2_DriverLoad_Index.pdf
│
├── Fig_Type_5_Robustness/
│   ├── Case1_Authority_Disturbance.pdf
│   ├── Case1_Robustness_PhasePlane.pdf
│   ├── Case1_Robustness_LateralError.pdf
│   └── Case1_Robustness_HeadingError.pdf
│
└── Tables_Statistics/
    ├── Table_Overall_Performance.xlsx
    ├── Table_Safety_Metrics.xlsx
    └── Table_Intent_Consistency.xlsx
```

---

## 5. 对比方法设置

如果没有额外说明，请至少考虑以下对比方法：

| Abbreviation | Method Description | Role |
|---|---|---|
| DO | Driver only | 体现单纯驾驶人控制下的安全性与操作负荷 |
| AO | Automation only | 体现单纯机器意图或自动化控制效果 |
| Fixed | Fixed authority shared control | 体现固定权限分配的局限性 |
| Ref-Auth | Reference authority without RL optimization | 体现双向信任生成的参考权限 |
| RL-Auth | RL-optimized authority without robust execution | 体现 RL 权限优化效果 |
| Proposed | Proposed shared decision-making + RL authority + adaptive robust MPC | 本文完整方法 |

如果实验中已有不同命名方式，请保持论文全文命名一致。

---

## 6. 场景总览表

请首先用一个表格描述所有实验场景。

表格格式如下：

| Case No. | Scenario Type | Scenario Illustration | Driver Intention | Automation Intention | Conflict / Consistency Feature | Purpose |
|---|---|---|---|---|---|---|

要求：

1. 至少包括人机意图冲突场景和人机意图一致场景；
2. 每个场景用最简洁凝练的语言描述；
3. 场景图应体现关键车辆、道路结构、障碍车和主车位置；
4. 每个场景应明确驾驶人意图、机器意图及二者差异；
5. 每个场景应说明用于验证哪一个实验问题。

---

## 7. 第一类图：不同场景下的轨迹图

### 7.1 该类图的核心问题

当前轨迹图存在的问题是：图例遮挡核心区域，轨迹太短，人机轨迹差异不明显，不同方法之间的差异也不明显。因此必须重画。

轨迹图必须明确回答：

1. 人和机器到底冲突在哪里？
2. 本文共享决策到底融合了什么？
3. 本文最终控制轨迹相比其他方法好在哪里？

如果这三个问题在图上一眼看不出来，该图必须重新设计。

### 7.2 保存路径

```text
Experiment_Figures/Fig_Type_1_Trajectory/
```

### 7.3 强制拆图要求

不要把“人机意图展示”和“不同方法对比”混在一张图中。Case 1 至少应拆成三张图。

#### 图 A：人机意图冲突图

文件名：

```text
Case1_HM_Intention_Conflict.pdf
```

目的：

证明驾驶人意图和机器意图不同。

应包含：

1. Human intention trajectory；
2. Machine intention trajectory；
3. Obstacle vehicle；
4. Conflict region；
5. Vehicle snapshots；
6. Motion direction arrow；
7. Key time markers。

不要加入太多控制方法，不要把 Proposed、Fixed、RL-Auth 全放进来。

#### 图 B：共享参考生成图

文件名：

```text
Case1_Shared_Reference_Generation.pdf
```

目的：

证明本文方法能够融合人机意图，生成合理共享参考。

应包含：

1. Human intention trajectory；
2. Machine intention trajectory；
3. Trust-guided shared reference；
4. RL-optimized shared reference；
5. Obstacle vehicle；
6. Conflict region。

重点体现 shared reference 是否处于合理、安全的位置。

#### 图 C：不同方法执行轨迹对比图

文件名：

```text
Case1_Method_Execution_Comparison.pdf
```

目的：

证明本文方法的最终执行轨迹优于其他方法。

应包含：

1. Driver only；
2. Automation only；
3. Fixed authority；
4. Reference authority only；
5. Proposed。

重点体现安全距离、换道平滑性、避障效果和轨迹执行质量。

### 7.4 轨迹图基本要求

1. 只画关键时间段内的局部轨迹，不需要画全程；
2. 关键时间段必须覆盖：
   - 冲突开始前；
   - 冲突发生时；
   - 权限调整时；
   - 共享参考生成时；
   - 控制执行时；
   - 冲突解除后；
3. 背景车辆使用透明度递减表示时间顺序；
4. 主车不同方法只需要画出轨迹即可；
5. 轨迹颜色和线型要清晰区分；
6. 图例不能遮挡车辆、轨迹、冲突区域和关键时间标注；
7. 应使用浅灰色道路背景和浅灰色车道线，避免深色背景压制轨迹；
8. 加入箭头表示车辆运动方向；
9. 加入关键时间点，例如：
   - `t1: conflict begins`
   - `t2: authority changes`
   - `t3: shared reference generated`
   - `t4: risk resolved`

### 7.5 推荐线型

| Curve | Meaning | Suggested Style |
|---|---|---|
| Human intention | 驾驶人意图轨迹 | dashed line |
| Machine intention | 机器意图轨迹 | dash-dot line |
| Trust-guided shared reference | 双向信任生成的共享参考 | solid line |
| RL-optimized shared reference | RL 优化后的共享参考 | dotted or thick solid line |
| Proposed execution | 本文最终执行轨迹 | thick solid line |
| Obstacle vehicle | 障碍车轨迹或位置序列 | gray transparent vehicle snapshots |

### 7.6 不合格轨迹图判据

如果出现以下任一问题，该轨迹图视为不合格，必须重画：

1. 图例遮挡车辆、道路、轨迹或冲突区域；
2. 人机意图轨迹几乎重合，看不出冲突；
3. 共享参考轨迹和执行轨迹看不出差异；
4. 不同方法轨迹差异不明显；
5. 轨迹时间窗口太短，看不出完整机动过程；
6. 道路背景过重，轨迹不突出；
7. 图内标题过长或使用难懂缩写；
8. 没有关键时间标注；
9. 没有运动方向箭头；
10. 无法支撑论文创新点。

---

## 8. 第二类图：不同场景下的权限变化图

### 8.1 图组目标

该类图用于展示在人机双向信任变化下，参考权限和 RL 优化权限如何随时间变化。

### 8.2 保存路径

```text
Experiment_Figures/Fig_Type_2_Authority/
```

### 8.3 出图要求

1. 时间窗口应与第一类轨迹图的关键时间段一致；
2. 横轴为时间；
3. 纵轴为权限或信任度；
4. 曲线包括：
   - 参考权限曲线；
   - RL 优化权限曲线；
   - 机器对驾驶人的信任度曲线；
   - 驾驶人对机器的信任度曲线；
5. 图例不得遮挡权限突变点、信任变化点和核心曲线；
6. 如果同一张图中曲线过多，可以拆成两张图：
   - 权限曲线图；
   - 双向信任曲线图。

### 8.4 推荐变量

| Variable | Meaning |
|---|---|
| `lambda_ref` | 基于双向信任生成的参考权限 |
| `lambda_RL` | RL 优化后的权限 |
| `T_m2h` | 机器对驾驶人的信任度 |
| `T_h2m` | 驾驶人对机器的信任度 |

### 8.5 需要体现的结论

1. 当风险升高或机器对驾驶人的信任度降低时，驾驶人权限下降；
2. 参考权限可能存在突变或不够平滑；
3. RL 优化后的权限更加平滑；
4. RL 优化权限能够形成更合理的未来权限序列；
5. 双向信任变化能够有效驱动共享权限分配。

---

## 9. 第三类图：侧向动力学性能图

### 9.1 图组目标

该类图用于验证不同对比方法下车辆侧向稳定性和横向动力学性能。

### 9.2 保存路径

```text
Experiment_Figures/Fig_Type_3_Lateral_Dynamics/
```

### 9.3 质心侧偏角-横摆角速度相平面图

要求：

1. 横轴为质心侧偏角；
2. 纵轴为横摆角速度；
3. 对比不同控制方法；
4. 可绘制稳定边界；
5. 图例不得遮挡稳定边界和主要相轨迹；
6. 相轨迹应清晰体现不同方法的稳定性差异。

需要体现：

1. 本文方法的相轨迹更集中；
2. 本文方法更接近稳定区域中心；
3. 本文方法具有更好的横向稳定性；
4. 其他方法可能出现更大的横摆响应或侧偏角偏移。

### 9.4 侧向加速度时序图

要求：

1. 横轴为时间；
2. 纵轴为侧向加速度；
3. 对比不同控制方法；
4. 可标注舒适性或稳定性阈值；
5. 图例不得遮挡峰值位置和阈值线。

需要体现：

1. 本文方法侧向加速度峰值更小；
2. 本文方法侧向加速度变化更平滑；
3. 本文方法具有更好的稳定性和舒适性。

---

## 10. 第四类图：驾驶人操作负荷图

### 10.1 图组目标

该类图用于验证本文方法在保证安全性的同时，是否能够降低驾驶人操作负荷。

### 10.2 保存路径

```text
Experiment_Figures/Fig_Type_4_Driver_Operation/
```

### 10.3 出图要求

1. 主要用于人机意图一致场景；
2. 可绘制驾驶人转向角-时间图；
3. 可绘制驾驶人转向角-车速图；
4. 可绘制驾驶人转向角速度或驾驶人输入能量；
5. 对比不同方法；
6. 图例不得遮挡驾驶人操作峰值和关键变化区间。

### 10.4 推荐变量

| Variable | Meaning |
|---|---|
| `delta_d` | 驾驶人转向角 |
| `d_delta_d` | 驾驶人转向角速度 |
| `v_x` | 车速 |
| `E_driver` | 驾驶人输入能量或操作负荷指标 |

### 10.5 需要体现的结论

1. 本文方法可以降低驾驶人转向角峰值；
2. 本文方法可以降低驾驶人转向频繁程度；
3. 本文方法可以降低驾驶人输入能量；
4. 本文方法在保证安全性和稳定性的同时降低驾驶负荷。

---

## 11. 第五类图：抗干扰性与鲁棒性图

### 11.1 图组目标

该类图用于验证当 RL 优化后的权限序列存在误差或随机扰动时，adaptive robust MPC 是否仍能保证车辆稳定性和跟踪性能。

### 11.2 保存路径

```text
Experiment_Figures/Fig_Type_5_Robustness/
```

### 11.3 扰动设置

请在 RL 优化后的权限基础上加入随机扰动或阶跃扰动，例如：

```text
lambda_disturbed = lambda_RL + Delta_lambda
```

其中，`Delta_lambda` 可以为：

1. 有界随机扰动；
2. 阶跃扰动；
3. 正弦扰动；
4. 短时脉冲扰动。

扰动后的权限应限制在合理范围内，例如 `[0, 1]`。

### 11.4 推荐图形

| Figure | Description |
|---|---|
| Authority disturbance profile | 展示 RL 权限和扰动后权限 |
| Phase-plane stability plot | 展示扰动下车辆稳定性 |
| Lateral acceleration plot | 展示扰动下侧向加速度 |
| Lateral error plot | 展示扰动下横向跟踪误差 |
| Heading error plot | 展示扰动下航向误差 |

### 11.5 需要体现的结论

1. 权限扰动会降低车辆稳定性或跟踪性能；
2. adaptive robust MPC 能够抑制权限误差导致的不稳定；
3. 本文方法可以减少权限误差造成的危险；
4. 本文方法在权限扰动下仍具有较好的鲁棒性。

---

## 12. 综合统计表

### 12.1 表格目标

该表用于对所有场景和所有方法进行综合统计比较，支撑本文方法在安全性、效率、稳定性、意图一致性和驾驶负荷方面的整体优势。

### 12.2 保存路径

```text
Experiment_Figures/Tables_Statistics/
```

### 12.3 推荐指标

| Metric | Meaning | Preferred Trend |
|---|---|---|
| Minimum distance | 主车与障碍车的最小欧式距离 | Larger is better |
| TTC | Time-to-collision | Larger is better |
| Travel efficiency | 出行效率，例如完成时间或平均速度 | Higher efficiency is better |
| Steering consistency | 人机转向一致性 | Larger is better |
| Trajectory intention consistency | 轨迹意图一致性 | Larger is better |
| Human-machine conflict intensity | 人机冲突强度 | Smaller is better |
| RMS lateral error | 横向误差均方根 | Smaller is better |
| Peak lateral acceleration | 侧向加速度峰值 | Smaller is better |
| Driver steering effort | 驾驶人转向负荷 | Smaller is better |
| Yaw rate peak | 横摆角速度峰值 | Smaller is better |
| Sideslip angle peak | 质心侧偏角峰值 | Smaller is better |

### 12.4 表格格式

| Scenario | Method | Min. Distance | TTC | Efficiency | Steering Consistency | Trajectory Consistency | Conflict Intensity | RMS Lateral Error | Peak Lateral Acc. | Driver Effort |
|---|---|---|---|---|---|---|---|---|---|---|

---

## 13. 每张图的输出信息要求

请最终用如下表格总结所有推荐图：

| Figure/Table No. | File Name | Folder | Scenario | Compared Methods | Signals / Variables | X-axis | Y-axis | Expected Trend | Supported Claim | Main Text or Supplement |
|---|---|---|---|---|---|---|---|---|---|---|

要求：

1. 每张图都要单独列出；
2. 不要只写“轨迹图”或“权限图”；
3. 要具体到文件名；
4. 要说明该图支撑哪一个论文结论；
5. 要说明推荐放在主文还是补充材料。

---

## 14. 推荐论文实验章节组织结构

请进一步给出推荐的实验章节结构，例如：

```text
VI. Experiments and Results

A. Experimental Setup and Compared Methods
B. Scenario Description and Human-Machine Intention Conflicts
C. Evaluation of Shared Intention Generation and Trajectory Execution
D. Evaluation of Trust-Guided and RL-Optimized Authority Allocation
E. Evaluation of Lateral Stability and Driver Operation Burden
F. Robustness Analysis Under Authority Disturbances
G. Quantitative Comparison and Discussion
```

要求说明：

1. 每一小节应该放哪些图；
2. 每一小节主要证明什么；
3. 每一小节对应哪一个创新点。

---

## 15. 英文图题要求

请为每张图给出 IEEE Transactions 风格的英文图题。

图题要求：

1. 简洁；
2. 准确；
3. 不要过度宣传；
4. 能体现实验变量和场景；
5. 与论文技术逻辑一致。

示例：

```text
Fig. X. Trajectory comparison under human-machine intention conflict in Case 1.
Fig. X. Reference and RL-optimized authority profiles in Case 1.
Fig. X. Phase-plane comparison of sideslip angle and yaw rate under different methods.
Fig. X. Lateral acceleration responses under different shared control strategies.
Fig. X. Robustness evaluation under perturbed authority allocation.
```

---

## 16. 主文与补充材料优先级

### 16.1 必须放主文的核心图

1. 场景总览表；
2. 人机意图冲突轨迹图；
3. 共享参考生成图；
4. 不同方法执行轨迹对比图；
5. 参考权限与 RL 优化权限对比图；
6. 相平面稳定性图；
7. 侧向加速度图；
8. 综合统计表。

### 16.2 可放主文或补充材料的图

1. 驾驶人转向角-车速图；
2. 驾驶人转向角速度图；
3. 权限预测序列图；
4. 横向误差时序图；
5. 航向误差时序图。

### 16.3 适合放补充材料的图

1. 每个 Case 的完整轨迹；
2. 所有扰动形式对比；
3. 所有指标的完整时序曲线；
4. 所有方法的额外消融结果。

---

## 17. 当前错误图的问题总结

当前生成的轨迹图主要存在以下问题：

1. **图例遮挡严重**  
   图例位于道路中心区域，遮挡车辆、车道线和轨迹。

2. **人机轨迹无法对比**  
   驾驶人意图、机器意图、共享参考和执行轨迹之间的差异不明显。

3. **不同方法差异不明显**  
   图中更像是本文内部变量展示，而不是对比不同方法性能。

4. **轨迹时间窗口太短**  
   没有覆盖冲突发生、权限调整、共享参考生成、控制执行和冲突解除的完整过程。

5. **道路背景过重**  
   深色道路背景和亮色车道线抢占视觉注意力，轨迹反而不突出。

6. **标题不符合 IEEE 风格**  
   图内标题过长，且使用 `H/M/Shared = R/S/S` 这类难懂缩写，不适合作为正式论文图。

7. **缺少关键事件标注**  
   没有明确标注 conflict begins、authority changes、shared reference generated、risk resolved 等关键时间点。

8. **缺少运动方向表达**  
   没有箭头或时间递进标注，读者难以理解车辆动态过程。

---

## 18. 强制修改清单

请严格按以下要求修改当前出图：

1. Remove the large in-figure legend. Place the legend outside the plotting area, preferably above the figure in a single row or below the figure.
2. Do not let the legend overlap with road lanes, vehicle snapshots, trajectory curves, obstacle vehicles, or conflict regions.
3. Separate intention visualization and method comparison into different figures:
   - Figure A: human intention vs. machine intention;
   - Figure B: human/machine intention vs. shared reference;
   - Figure C: different control method execution trajectories.
4. Extend the plotted trajectory window to cover the complete key maneuver process, including conflict occurrence, authority adjustment, trajectory generation, control execution, and conflict resolution.
5. Use a light gray road background and light gray lane markings. Avoid dark road background if it weakens trajectory visibility.
6. Use distinctive line styles rather than relying only on colors:
   - Human intention: dashed line;
   - Machine intention: dash-dot line;
   - Shared reference: solid line;
   - RL-optimized reference: dotted or thick solid line;
   - Proposed execution: thick solid line.
7. Add key time markers such as `t1`, `t2`, `t3`, and `t4` to indicate conflict occurrence, decision update, authority change, and conflict resolution.
8. Add vehicle snapshots with decreasing transparency to indicate temporal evolution.
9. Add arrows to indicate driving direction.
10. Use a concise IEEE-style caption instead of a long title inside the figure.
11. Make all axis labels, units, line widths, and fonts consistent with IEEE Transactions style.
12. Export each figure separately as PDF/EPS/SVG and PNG at 600 dpi.
13. Run an automatic or manual visual check after plotting. If legend overlap, unreadable trajectories, missing labels, or unclear method differences remain, revise the plotting code and regenerate the figure.

---

## 19. 最终交付要求

请最终输出以下内容：

1. 修改后的完整出图方案；
2. 每一类图对应的具体图片清单；
3. 每张图的横轴、纵轴、变量、图例、线型建议；
4. 每张图的保存路径和文件名；
5. 每张图支撑的论文结论；
6. 每张图的英文图题；
7. 推荐的实验章节组织结构；
8. 主文图和补充材料图的优先级；
9. MATLAB 或 Python 的统一绘图风格模板；
10. 所有生成图片的文件路径；
11. 对每张图进行自检，说明是否存在图例遮挡、轨迹不清晰、方法差异不明显等问题。

---

## 20. 最重要的判断标准

当前任务的核心不是“让图更漂亮”，而是**让图把论文创新点画出来**。

每张核心图都必须能回答：

```text
1. 人和机器到底冲突在哪里？
2. 本文共享决策到底融合了什么？
3. 本文最终控制轨迹相比其他方法好在哪里？
```

如果一张图不能回答上述问题，应重新设计，而不是继续微调颜色和线宽。
