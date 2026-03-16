# Phase A：Quaternion Interpretation 使用方式详解

本文档面向不熟悉深度学习工程的研究者，按「从零到跑通」的顺序说明每一步在做什么、如何操作、以及如何检查结果。

---

## 一、前置准备（环境与数据）

### 1.1 确认环境已安装

本项目基于 PyTorch + Lightning。请先确保已经完成：

- 创建并激活 conda 环境（例如 `gspl`）
- 安装 PyTorch（与 CUDA 版本匹配）
- 在项目根目录执行：`pip install -r requirements.txt`
- 若使用 gsplat 后端，按 README 安装 gsplat

**如何检查：** 在项目根目录打开终端，执行：

```bash
python -c "import torch; print(torch.cuda.is_available()); from internal.renderers.quaternion_interpretation_renderer import QuaternionInterpretationRenderer; print('OK')"
```

若输出 `True` 和 `OK`，说明环境与 Phase A 的 renderer 都可正常导入。

### 1.2 准备数据集（以 Blender/nerf_synthetic 为例）

Phase A 文档里以 **lego** 为例，数据集通常为 **nerf_synthetic**（Blender 格式）。

1. **下载数据集**  
   - 官方链接（README 中）：  
     https://drive.google.com/drive/folders/1JDdLGDruGNXWnM1eqY1FNL9PlStjaKWi  
   - 下载并解压后，得到类似 `nerf_synthetic/lego/` 的目录。

2. **目录结构要求**  
   解压后应至少包含：
   - `train/` 或 `images/`：训练图像
   - `test/` 或 `val/`：测试/验证图像（可选，用于 validate）
   - `transforms_train.json`、`transforms_test.json`（Blender 格式）

3. **记住数据集路径**  
   后续命令里的 `--data.path` 要指向「场景」目录，即包含上述内容的**那一层**。例如：
   - 若解压到 `D:\datasets\nerf_synthetic\lego`，则 `--data.path` 为：
     - `D:\datasets\nerf_synthetic\lego`
   - 或使用相对路径（在项目根目录下）：`datasets/nerf_synthetic/lego`

**建议：** 在项目根目录下建一个 `datasets` 文件夹，把 `nerf_synthetic` 放进去，这样路径为 `datasets/nerf_synthetic/lego`，便于书写和复现。

---

## 二、配置文件是做什么的（理解「叠加」）

训练命令里会出现多个 `--config xxx.yaml`。它们会**按顺序叠加**：后面的配置会覆盖前面同名的项，不会改动的项保持不变。

- **configs/blender.yaml**  
  针对 Blender 数据集的通用设置（如验证频率、数据解析方式等）。

- **configs/gsplat_v1.yaml**  
  指定使用 gsplat 的某一版 renderer 等（若你 baseline 用 diff-gaussian-rasterization，可以不用 gsplat，改用与 baseline 一致的配置即可）。

- **configs/quaternion_phase_a.yaml**  
  只做一件事：把「当前使用的 renderer」换成 `QuaternionInterpretationRenderer`，其它（数据、loss、densification、Gaussian 模型等）都不改。

因此：

- **Baseline 训练**：不加 `quaternion_phase_a.yaml`，用你平时用的 config 即可。
- **Phase A 训练**：在 baseline 的 config 后面**再**加一条 `--config configs/quaternion_phase_a.yaml`，就只把「渲染时的颜色解释」换成 quaternion 那一套，其余完全一致。

这样设计是为了「最小侵入」：只改渲染解释层，方便对比。

---

## 三、训练（fit）— 一步一步来

### 3.1 命令在做什么

- `python main.py fit`：启动「训练」子命令。
- `--config ...`：指定配置文件（可多个，按顺序叠加）。
- `--data.path ...`：数据集路径（必须指向具体场景，如 lego）。
- `-n ...`：实验名称，会作为输出目录名的一部分（如 `outputs/lego_quaternion_phase_a`）。
- `--max_steps 10000`：最多训练 10000 步（与 Phase A 文档一致）。
- `--save_iterations "[101,501,1001,3001,5001]"`：在这些步数**结束时**额外保存一次 checkpoint；**训练结束**时还会自动再保存一次（即 10000 步的最终 ckpt）。

### 3.2 在 Windows 下怎么写命令

**方式一：一行写完整（推荐，避免换行符问题）**

在 **PowerShell** 或 **命令提示符（cmd）** 中，先进入项目根目录，再执行（把 `DATASET_PATH` 换成你的实际路径）：

```powershell
cd D:\Programming\Python\3DGS\platform_new\gaussian-splatting-lightning
```

```powershell
python main.py fit --config configs/blender.yaml --config configs/gsplat_v1.yaml --config configs/quaternion_phase_a.yaml --data.path datasets/nerf_synthetic/lego -n lego_quaternion_phase_a --max_steps 10000 --save_iterations "[101,501,1001,3001,5001]"
```

若你的 baseline 从不用 gsplat_v1，而是用别的（例如只用 `configs/blender.yaml`），则去掉 `configs/gsplat_v1.yaml`，保持与 baseline 一致，只多加 Phase A 的 config：

```powershell
python main.py fit --config configs/blender.yaml --config configs/quaternion_phase_a.yaml --data.path datasets/nerf_synthetic/lego -n lego_quaternion_phase_a --max_steps 10000 --save_iterations "[101,501,1001,3001,5001]"
```

**方式二：多行书写（PowerShell）**

在 PowerShell 中，行末用反引号 `` ` `` 表示「命令在下一行继续」：

```powershell
python main.py fit `
  --config configs/blender.yaml `
  --config configs/gsplat_v1.yaml `
  --config configs/quaternion_phase_a.yaml `
  --data.path datasets/nerf_synthetic/lego `
  -n lego_quaternion_phase_a `
  --max_steps 10000 `
  --save_iterations "[101,501,1001,3001,5001]"
```

**注意：**  
- `--save_iterations` 的值必须是**列表**格式：`"[101,501,1001,3001,5001]"`（带方括号、逗号无空格或保留空格均可）。  
- 路径中的反斜杠：在 PowerShell 中可用 `/` 或 `\`；若路径含空格，请用双引号包起来，如 `"--data.path=D:\my data\nerf_synthetic\lego"`。

### 3.3 训练时终端会看到什么

- 启动时会打印：数据加载、模型/渲染器类型、optimizer 等。
- 训练过程中会有进度条，并显示例如：
  - `train/loss`：当前 step 的 loss
  - `train/rgb_diff`、`train/ssim` 等（若使用 VanillaMetrics）
  - 有时会有 `gaussians_count` 等
- 在 101、501、1001、3001、5001 步结束时会保存 checkpoint，并打印类似：`Checkpoint saved to ...`。
- 训练到 10000 步结束后，会再保存一次「最终」checkpoint。

若出现 **NaN / Inf** 或 loss 长时间不下降、报错，请先记下完整报错信息，并对照文档最后的「排查优先级」检查。

### 3.4 输出目录在哪里、长什么样

- 默认情况下，输出根目录为项目下的 **`outputs`**（可由 `--output` 修改）。
- 实验名由 `-n` 指定，例如 `lego_quaternion_phase_a`，则主要目录为：
  - **`outputs/lego_quaternion_phase_a/`**
- 其下通常会有：
  - **`checkpoints/`**：所有保存的 `.ckpt` 文件，例如：
    - `epoch=0-step=101.ckpt`
    - `epoch=0-step=501.ckpt`
    - …
    - `epoch=0-step=10000.ckpt`（最终）
  - 可能还有 `lightning_logs/version_0/`（TensorBoard 日志、config 备份等）。

**如何确认：** 训练结束后在资源管理器中打开 `outputs/lego_quaternion_phase_a/checkpoints/`，应能看到多个 `epoch=*-step=*.ckpt` 文件。

---

## 四、验证（validate）— 对每个 checkpoint 做一次评估

### 4.1 validate 在做什么

- 加载**指定**的 checkpoint（包含 Gaussian 参数和 renderer 配置）。
- 在验证集（或测试集）上渲染图像。
- 计算指标（PSNR、SSIM、LPIPS 等）并可选地保存渲染图。

Phase A 要求对 101、501、1001、3001、5001、10000 这几个 step 的 checkpoint 都跑一遍 validate，以便和 baseline 逐 step 对比。

### 4.2 命令含义

- `python main.py validate`：启动「验证」子命令。
- `--config ...`：**必须与训练时一致**（同样叠加 blender + gsplat_v1 + quaternion_phase_a），这样 renderer 类型才对。
- `--data.path ...`：与训练相同的数据路径（验证集会从同一数据集里按 parser 划分）。
- `--ckpt_path ...`：指向某一个 `.ckpt` 文件（或指向包含 `.ckpt` 的目录，程序会自动选一个）。
- `--model.save_val_output true`：把渲染出的图像存成文件。
- `--model.save_val_metrics true`：把指标写入 CSV 等。
- `--save_val`：启用「保存验证结果」。
- `-n ...`：验证实验的名字；可以按 step 区分，例如 `lego_quaternion_phase_a_step10000_eval`，这样不同 step 的验证结果存在不同子目录，不会互相覆盖。

### 4.3 对每个 checkpoint 各运行一次

下面以 step 10000 为例；其它 step 只需把 `--ckpt_path` 和 `-n` 里的步数改掉即可。

**Step 10000 的示例（一条命令）：**

```powershell
python main.py validate --config configs/blender.yaml --config configs/gsplat_v1.yaml --config configs/quaternion_phase_a.yaml --data.path datasets/nerf_synthetic/lego --ckpt_path outputs/lego_quaternion_phase_a/checkpoints/epoch=0-step=10000.ckpt --model.save_val_output true --model.save_val_metrics true --save_val -n lego_quaternion_phase_a_step10000_eval
```

**注意：**  
- `epoch=0-step=10000.ckpt` 中的数字取决于 Lightning 实际保存的文件名；若你的文件名是 `epoch=1-step=10000.ckpt`，就改成对应名字。  
- 若不确定文件名，可到 `outputs/lego_quaternion_phase_a/checkpoints/` 下查看实际存在的 `.ckpt` 文件名。

**其它 step 的模板（替换 STEP 和 步数）：**

- STEP=101：  
  `--ckpt_path outputs/lego_quaternion_phase_a/checkpoints/epoch=0-step=101.ckpt`  
  `-n lego_quaternion_phase_a_step00101_eval`
- STEP=501：  
  `--ckpt_path .../epoch=0-step=501.ckpt`  
  `-n lego_quaternion_phase_a_step00501_eval`
- 1001、3001、5001 同理。

也可以写一个简单脚本，循环上述命令，依次传入不同 step 和对应的 ckpt 路径。

### 4.4 验证结果存在哪里

- 默认在 **`outputs/`** 下，按 `-n` 的名字再建一层目录，例如：
  - `outputs/lego_quaternion_phase_a_step10000_eval/`
- 其中会有验证阶段保存的渲染图、metrics CSV 等（具体结构以项目实际为准，可在该目录下查看）。

---

## 五、和 Baseline 对比时要做的事

1. **用相同数据、相同 config（仅不加 quaternion_phase_a）** 训练一版 baseline，并保存相同 step 的 checkpoint（101, 501, 1001, 3001, 5001, 10000）。
2. 对 baseline 的每个 checkpoint 同样跑一遍 validate（config 不加 quaternion_phase_a）。
3. 对比同一 step 下：
   - Phase A 与 baseline 的 **PSNR / SSIM / LPIPS** 是否同量级（Phase A 不要求更好，但不能异常变差）。
   - 渲染图是否有全黑、全白、严重色偏、明显糊掉等。

若 Phase A 与 baseline 指标接近、且没有明显视觉异常，即可认为 Phase A「可嵌入性」验证通过，可以进入 Phase B。

---

## 六、可选：用脚本跑训练和验证（experiments）

项目里有 `experiments/`，可以用「实验名」统一管理命令。

### 6.1 在 cases 里加一个 Phase A 的 case

编辑 **`experiments/cases.py`**：

- 复制一份已有的 case（例如 `lego_baseline`），改名为 `lego_quaternion_phase_a`。
- 修改：
  - `train_exp_name`：例如 `"lego_quaternion_phase_a"`。
  - `train_configs`：在原有元组末尾加上 `"configs/quaternion_phase_a.yaml"`，例如：
    - `("configs/blender.yaml", "configs/gsplat_v1.yaml", "configs/quaternion_phase_a.yaml")`
  - `data_path`、`max_steps`、`save_iterations` 与文档一致即可。

保存后，在项目根目录执行：

```powershell
python experiments/run_fit.py --case lego_quaternion_phase_a
```

即会按 cases 里该 case 的 config 与路径启动训练。

### 6.2 用 run_validate 跑验证

**注意：** `experiments/run_validate.py` 会从 **`outputs/<train_exp_name>/lightning_logs/version_0/config.yaml`** 读取训练时的 config。若你训练时没有生成该路径的 config，可能找不到文件；此时用「第四节」里手写 `--config` 的方式更稳妥。

若你希望用脚本验证，需要：

- 在 `cases.py` 里有一个 case，其 `train_exp_name` 为 `lego_quaternion_phase_a`，且 `find_training_config` 能找到该实验的 config（即训练时确实写入了 `outputs/lego_quaternion_phase_a/lightning_logs/version_0/config.yaml`）。
- 然后执行：
  - 指定 step：  
    `python experiments/run_validate.py --case lego_quaternion_phase_a --step 10000`
  - 不指定 step（用最新 ckpt）：  
    `python experiments/run_validate.py --case lego_quaternion_phase_a`

若脚本报错「找不到 config」或「找不到 checkpoint」，直接改用第四节中的手写 `python main.py validate ...` 即可。

---

## 七、常见问题与排查顺序

1. **训练/验证时报错「找不到模块」或「QuaternionInterpretationRenderer」**  
   - 确认在**项目根目录**下执行命令。  
   - 确认已添加并保存了 `internal/renderers/quaternion_interpretation_renderer.py`，且 `internal/renderers/__init__.py` 中已导出 `QuaternionInterpretationRenderer`。

2. **validate 时渲染结果全黑/全白或颜色明显不对**  
   - 确认 validate 时使用的 **config 与训练完全一致**（包含 `quaternion_phase_a.yaml`）。  
   - 确认没有误用 baseline 的 checkpoint 配 Phase A 的 config（或反过来）。

3. **找不到 checkpoint 文件**  
   - 到 `outputs/lego_quaternion_phase_a/checkpoints/` 下看实际文件名（可能是 `epoch=0-step=10000.ckpt` 或 `epoch=1-step=10000.ckpt` 等），把 `--ckpt_path` 写成**完整路径**指向该文件。

4. **loss 出现 NaN**  
   - 先确认数据路径正确、数据能正常加载。  
   - 确认 `eval_sh` 与 quaternion 转换的维度、设备一致；Phase A 中 RGB→quat→RGB 应为恒等，若仍 NaN，可检查是否有除零（如 `dir_pp_norm` 已用 `clamp_min(eps)`）。

5. **与 baseline 指标差很多**  
   - 确认 baseline 与 Phase A 使用**相同** `--data.path`、相同数据划分。  
   - 确认 baseline 没有多/少叠加 config（例如 baseline 若不用 gsplat_v1，Phase A 也不应多这一层）。

---

## 八、Phase A 成功判据（自查清单）

- [ ] 训练完整跑完 10000 步，无崩溃、无 NaN。  
- [ ] `outputs/lego_quaternion_phase_a/checkpoints/` 中有 101、501、1001、3001、5001、10000 的 checkpoint。  
- [ ] 对上述每个 step 都跑过 validate，且能正常生成渲染图和 metrics。  
- [ ] 渲染图看起来正常（无全黑/全白/严重色偏）。  
- [ ] PSNR/SSIM/LPIPS 与 baseline 同量级，无异常劣化。  

若全部打勾，可以认为 Phase A 通过，可进入 Phase B（替换外观参数化等）。

---

## 九、命令速查表（复制即用）

**训练（请把 `datasets/nerf_synthetic/lego` 换成你的数据路径）：**

```powershell
python main.py fit --config configs/blender.yaml --config configs/gsplat_v1.yaml --config configs/quaternion_phase_a.yaml --data.path datasets/nerf_synthetic/lego -n lego_quaternion_phase_a --max_steps 10000 --save_iterations "[101,501,1001,3001,5001]"
```

**验证（请把 step 和 ckpt 文件名按实际修改）：**

```powershell
python main.py validate --config configs/blender.yaml --config configs/gsplat_v1.yaml --config configs/quaternion_phase_a.yaml --data.path datasets/nerf_synthetic/lego --ckpt_path outputs/lego_quaternion_phase_a/checkpoints/epoch=0-step=10000.ckpt --model.save_val_output true --model.save_val_metrics true --save_val -n lego_quaternion_phase_a_step10000_eval
```

如果你愿意，我可以再根据你当前的数据路径和 baseline 配置，帮你改成一版「可直接在你机器上运行」的完整命令（训练 + 多条验证）。
