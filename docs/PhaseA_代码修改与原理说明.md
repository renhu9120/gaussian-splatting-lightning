# Phase A 代码修改与原理说明

本文档说明：Phase A 在**原理上**是什么、在**代码上**具体改了什么、以及为什么和原版在数值上应当没有差别。

---

## 一、Phase A 的目标与原理

### 1.1 目标

- **不追求**立刻超过 baseline，只做「可嵌入性」验证。
- 验证：**纯四元数解释层**能否无故障地接入当前 3DGS 的 训练 → 渲染 → 验证 链路，且训练收敛、指标与 baseline 同量级。

### 1.2 形式上的约定：Pure Quaternion 颜色表示

约定颜色用**纯四元数**表示：

- **q = 0 + r·i + g·j + b·k**
- 实部为 0，三个虚部对应 RGB；不新增通道、不改 SH 体系。
- Phase A 里：先把 Python 侧算出的 **RGB 解释为** 上述纯四元数，再**从四元数取回 RGB** 喂给光栅化器。  
  也就是说，数据流是：**SH → RGB → (当作) 四元数 → 再取回 RGB → rasterizer**。

### 1.3 为何和原版「理应没有差别」

- 我们**没有**改：Gaussian 参数定义、SH 参数、loss、densification、几何、rasterizer 的 CUDA 核心。
- **唯一**多出来的一步是：**RGB → 纯四元数 → 再取回 RGB**。  
  在代码里：
  - `rgb_to_pure_quaternion(rgb)`：把 `[r,g,b]` 变成 `[0,r,g,b]`（四元数 w,x,y,z）；
  - `pure_quaternion_to_rgb(quat)`：把 `[w,x,y,z]` 取 `[x,y,z]` 作为 RGB。
- 因此 **RGB → quat → RGB** 在数值上就是**恒等**：进去是 (r,g,b)，出来还是 (r,g,b)。  
  所以 Phase A 只是「多走了一层四元数形式」，**不改变传给 rasterizer 的颜色数值**，理论上和「原版 VanillaRenderer 且在 Python 里做 SH→RGB 再 colors_precomp」应当一致。

---

## 二、代码上做了哪些修改（按文件）

下面按「文件」列出所有与 Phase A 相关的修改，并说明各自作用。

---

### 2.1 核心逻辑：新增 Renderer（唯一影响前向的改动）

**文件**：`internal/renderers/quaternion_interpretation_renderer.py`（全新）

**作用**：实现 Phase A 的「只改颜色解释层、其余不变」的 renderer。

**原理与实现要点**：

1. **继承 VanillaRenderer**  
   几何、光栅化设置、means/opacity/scales/rotations 等全部沿用 VanillaRenderer 的逻辑。

2. **强制在 Python 里做 SH→RGB**  
   - 在 `__init__` 里写死 `convert_SHs_python=True` 传给父类。  
   - 这样颜色一定在 Python 侧用 `eval_sh` 算成 RGB，方便在中间插入「四元数解释」这一步。

3. **两个静态方法（纯形式，数值恒等）**  
   - `rgb_to_pure_quaternion(rgb)`：输入 `[N,3]` 的 RGB，输出 `[N,4]`，布局为 `[w,x,y,z] = [0, r, g, b]`。  
   - `pure_quaternion_to_rgb(quat)`：输入 `[N,4]` 的 quat，输出 `quat[:, 1:4]`，即 `[N,3]` 的 RGB。  
   因此 `pure_quaternion_to_rgb(rgb_to_pure_quaternion(rgb)) == rgb`，只是多了一层「当作四元数」的形式。

4. **forward 里唯一多出来的三步（约 111–128 行）**  
   - 用和 VanillaRenderer 相同的方式：`eval_sh` + `clamp_min(sh2rgb + 0.5, 0.0)` 得到 `rgb`。  
   - 然后：
     - `quat = self.rgb_to_pure_quaternion(rgb)`
     - `rgb_from_quat = self.pure_quaternion_to_rgb(quat)`
     - `colors_precomp = rgb_from_quat`  
   - 因为 `rgb_from_quat` 与 `rgb` 数值相同，所以传给 rasterizer 的 `colors_precomp` 和「原版 VanillaRenderer + convert_SHs_python=True」完全一致。  
   其余（means3D、means2D、opacity、scales、rotations、rasterizer 调用、返回值）与 VanillaRenderer 一致。

5. **小差异**  
   - 对 `dir_pp` 做 `norm(dim=1, keepdim=True).clamp_min(self.eps)` 再除，避免除零；原版 VanillaRenderer 里没有 `clamp_min`。这是数值稳定性上的小改进，在正常数据下不改变结果。

**总结**：Phase A 在「前向计算」上的修改，**只**体现在这个新 renderer 的「SH→RGB 之后、进 rasterizer 之前」多了一步 RGB→quat→RGB 的形式转换，数值上恒等，因此和原版应无差别。

---

### 2.2 配置：让训练/验证使用 Phase A 的 Renderer

**文件**：`configs/quaternion_phase_a.yaml`（全新）

**内容**：只覆盖 `model.renderer`，指定使用 `QuaternionInterpretationRenderer`，并带少量 init 参数（如 `compute_cov3D_python`、`eps`、`debug_save_stats`）。

**作用**：  
- 训练或验证时，只要在原有 config 链上**追加**这一份 yaml，就会把「当前用的 renderer」换成 Phase A 的 renderer。  
- 不改 data、metric、density、gaussian 等，所以 Phase A 与 baseline 的差别**仅**在「用哪个 renderer」。

**原理**：配置叠加时，后加载的覆盖前面的同名字段；因此只改 renderer，其余保持 baseline 配置。

---

### 2.3 导出新 Renderer，供配置解析使用

**文件**：`internal/renderers/__init__.py`

**修改**：增加一行  
`from .quaternion_interpretation_renderer import QuaternionInterpretationRenderer`

**作用**：  
- 使 `configs/quaternion_phase_a.yaml` 里通过 `class_path: internal.renderers.quaternion_interpretation_renderer.QuaternionInterpretationRenderer` 能正确解析并实例化，无需改 CLI 或其它加载逻辑。

---

### 2.4 实验入口：用 case 驱动 Phase A 训练/验证

**文件**：`experiments/cases.py`

**修改**：新增 case，例如 `lego_quaternion_phase_a`（以及可选的 `chair_quaternion_phase_a` 等）。

**要点**：  
- `train_exp_name`：例如 `"lego_quaternion_phase_a"`，决定输出目录 `outputs/lego_quaternion_phase_a/`。  
- `train_configs`：显式写为 `("configs/blender.yaml", "configs/gsplat_v1.yaml", "configs/quaternion_phase_a.yaml")`。  
- 其它（`data_path`、`max_steps`、`save_iterations`）与 baseline 对齐。

**作用**：  
- `run_fit.py --case lego_quaternion_phase_a` 会按上述 `train_configs` 依次加 `--config`，从而在**不写死命令行**的情况下，用「baseline + quaternion_phase_a」配置跑 Phase A 训练。  
- 与 baseline 的**唯一**区别就是多了一个 `configs/quaternion_phase_a.yaml`，即多用了 Phase A 的 renderer。

**原理**：run_fit 根据 case 的 `train_configs` 拼出 `main.py fit --config ... --config ...`，因此「Phase A 修改」体现在 case 的 config 列表多了一项，而不是改 run_fit 或 main 的内部逻辑。

---

### 2.5 验证时没有「训练时保存的 config」时的回退逻辑

**文件**：`experiments/run_validate.py`

**修改**：在 `build_validate_command` 里，若 `find_training_config(case.train_exp_name)` 找不到文件（训练时没保存 config 或路径不一致），则不再报错，而是用 **case 的 `train_configs`** 拼出多组 `--config`，再调用 `main.py validate`。

**作用**：  
- 即使没有 `outputs/lego_quaternion_phase_a/.../config.yaml`，只要 case 里配置了 `train_configs`（含 `quaternion_phase_a.yaml`），验证仍会用与训练一致的 config 跑，包括使用 `QuaternionInterpretationRenderer`。

**原理**：验证必须和训练用同一套 config（尤其是同一 renderer）；用 case 的 `train_configs` 作为「训练时 config 的等价物」，保证行为一致。

---

### 2.6 其它辅助修改（可选）

- **experiments/common.py**：`find_training_config` 会尝试多个可能路径（如 `config.yaml`、`version_0/config.yaml`、`lightning_logs/version_0/config.yaml`），以便在 Lightning 保存 config 位置不一致时仍能找到；若找不到，就由上面的 run_validate 回退到 `train_configs`。
- **experiments/run_phase_a_validate_all.py**、**README/运行配置说明** 等：仅用于方便你「按 step 批量验证」和用 IDE 配置运行，不改变训练或验证的算法逻辑。

---

## 三、数据流对比（原版 vs Phase A）

**原版 VanillaRenderer（且 convert_SHs_python=True）**  
- SH 系数 → `eval_sh` → RGB → `clamp_min(sh2rgb+0.5, 0)` → `colors_precomp` → rasterizer → 图像。

**Phase A（QuaternionInterpretationRenderer）**  
- SH 系数 → `eval_sh` → RGB → `clamp_min(sh2rgb+0.5, 0)` → **rgb→quat→rgb（恒等）** → `colors_precomp` → rasterizer → 图像。

因此，**进入 rasterizer 的 colors_precomp 在两种情况下数值相同**；几何、opacity、scales、rotations 等也完全相同，所以 Phase A 和原版在数值上理应没有差别，只是多了一层「四元数形式」的包装，用于验证管线可嵌入性。

---

## 四、总结表

| 位置 | 类型 | 作用 |
|------|------|------|
| `internal/renderers/quaternion_interpretation_renderer.py` | 新增 | 唯一改变前向的代码：在 SH→RGB 之后插入 RGB→quat→RGB（恒等），再 colors_precomp。 |
| `configs/quaternion_phase_a.yaml` | 新增 | 仅覆盖 renderer 为 QuaternionInterpretationRenderer。 |
| `internal/renderers/__init__.py` | 一行 | 导出 QuaternionInterpretationRenderer，供 yaml class_path 使用。 |
| `experiments/cases.py` | 新增 case | 为 Phase A 指定 train_configs（含 quaternion_phase_a.yaml），从而用 run_fit 跑 Phase A。 |
| `experiments/run_validate.py` | 逻辑 | 找不到保存的 config 时，用 case.train_configs 拼 validate 的 --config，保证验证与训练一致。 |

**原理一句话**：Phase A 只在「颜色解释」上多了一层「先当纯四元数、再取回 RGB」的形式步骤，该步骤数值恒等，因此和原版应无差别，仅用于验证四元数接口可安全接入现有管线。
