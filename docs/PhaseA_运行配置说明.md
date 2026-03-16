# Phase A 运行配置说明（IDE 配置方式）

在 IDE 中通过 **运行配置（Run Configuration）** 运行 Phase A，无需在终端输入命令。  
请在你的 IDE 中新建「Python 运行配置」，并按下面表格填写各字段。

**工作目录（Working directory）** 必须为项目根目录（即包含 `main.py` 的目录）：

```
D:\Programming\Python\3DGS\platform_new\gaussian-splatting-lightning
```

---

## 一、Phase A 训练（lego）

在 IDE 中新建一条运行配置，填写：

| 字段 | 值 |
|------|-----|
| **Name** | `Phase A - Lego 训练` |
| **Script path** | `experiments/run_fit.py` 或 `D:\Programming\Python\3DGS\platform_new\gaussian-splatting-lightning\experiments\run_fit.py` |
| **Parameters** | `--case lego_quaternion_phase_a` |
| **Working directory** | `D:\Programming\Python\3DGS\platform_new\gaussian-splatting-lightning` |

说明：运行后会在 **`outputs/phase_a_lego/`** 下保存 checkpoint（`checkpoints/` 等）；目录名为「方法_数据集」。

---

## 二、Phase A 验证（所有 step：101, 501, 1001, 3001, 5001, 10000）

**前提**：已先完成「一、Phase A 训练」。与训练一致，**Script 仍用 run_validate.py**，只改 Parameters。

| 字段 | 值 |
|------|-----|
| **Name** | `Phase A - Lego 验证全部` |
| **Script path** | `experiments/run_validate.py`（与训练同脚本入口，仅参数不同） |
| **Parameters** | `--case lego_quaternion_phase_a --all_phase_a_steps` |
| **Working directory** | `D:\Programming\Python\3DGS\platform_new\gaussian-splatting-lightning` |

说明：会对 6 个 step 各跑一次 validate，结果写入 **`outputs/phase_a_lego/validate/step00101`**、**`outputs/phase_a_lego/validate/step00501`** 等，与训练同属一实验目录。

---

## 三、Phase A 验证（单个 step，可选）

若只想验证某一个 step，可单独建一条配置，把 `STEP` 换成 101 / 501 / 1001 / 3001 / 5001 / 10000：

| 字段 | 值 |
|------|-----|
| **Name** | `Phase A - Lego 验证 step=STEP`（例如 step=10000 时写 `Phase A - Lego 验证 step=10000`） |
| **Script path** | `experiments/run_validate.py` 或 `D:\Programming\Python\3DGS\platform_new\gaussian-splatting-lightning\experiments\run_validate.py` |
| **Parameters** | `--case lego_quaternion_phase_a --step STEP`（例如 `--case lego_quaternion_phase_a --step 10000`） |
| **Working directory** | `D:\Programming\Python\3DGS\platform_new\gaussian-splatting-lightning` |

---

## 四、可选：Phase A 冒烟测试（10 步）

用于快速检查配置是否正确，不会得到完整结果：

| 字段 | 值 |
|------|-----|
| **Name** | `Phase A - Lego 冒烟测试` |
| **Script path** | `experiments/run_fit.py` |
| **Parameters** | `--case lego_quaternion_phase_a --smoke` |
| **Working directory** | `D:\Programming\Python\3DGS\platform_new\gaussian-splatting-lightning` |

---

## 五、PyCharm 填写示例

在 **Run → Edit Configurations** 中：

1. 点击 **+** → **Python**。
2. **Name**：填入上表对应名称（如 `Phase A - Lego 训练`）。
3. **Script path**：点击文件夹图标，选择项目下的 `experiments/run_fit.py`（或对应脚本）。
4. **Parameters**：在「Parameters」输入框中填入 `--case lego_quaternion_phase_a`（或上表对应 Parameters）。
5. **Working directory**：选择项目根目录 `D:\Programming\Python\3DGS\platform_new\gaussian-splatting-lightning`。

---

## 六、VS Code / Cursor 填写示例

在 **运行和调试** 或 **Run and Debug** 中：

1. 点击「创建 launch.json」或编辑已有 `launch.json`。
2. 添加一条配置，例如 Phase A 训练：

```json
{
    "name": "Phase A - Lego 训练",
    "type": "debugpy",
    "request": "launch",
    "program": "${workspaceFolder}/experiments/run_fit.py",
    "args": ["--case", "lego_quaternion_phase_a"],
    "cwd": "${workspaceFolder}"
}
```

若你的解释器配置里用的是 `python` 而不是 `debugpy`，把 `"type": "debugpy"` 改成 `"type": "python"` 即可。

---

## 七、汇总表（复制用）

| 用途 | Name | Script path | Parameters | Working directory |
|------|------|-------------|------------|-------------------|
| Phase A 训练 | Phase A - Lego 训练 | experiments/run_fit.py | --case lego_quaternion_phase_a | 项目根目录 |
| Phase A 验证全部 | Phase A - Lego 验证全部 | experiments/run_validate.py | --case lego_quaternion_phase_a --all_phase_a_steps | 项目根目录 |
| Phase A 验证单步 | Phase A - Lego 验证 step=10000 | experiments/run_validate.py | --case lego_quaternion_phase_a --step 10000 | 项目根目录 |
| 冒烟测试 | Phase A - Lego 冒烟测试 | experiments/run_fit.py | --case lego_quaternion_phase_a --smoke | 项目根目录 |

**项目根目录** = `D:\Programming\Python\3DGS\platform_new\gaussian-splatting-lightning`
