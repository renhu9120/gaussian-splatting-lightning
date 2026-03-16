# Phase A：用 experiments 脚本训练与验证

数据集路径已在 `cases.py` 中设为 `D:\Programming\Python\3DGS\datasets\nerf_synthetic`，lego 等场景使用该路径下的子目录。

---

## 1. 训练 Phase A（lego）

在**项目根目录**下执行：

```bash
python experiments/run_fit.py --case lego_quaternion_phase_a
```

- 会自动加载：`configs/blender.yaml` + `configs/gsplat_v1.yaml` + `configs/quaternion_phase_a.yaml`
- 数据路径：`D:\Programming\Python\3DGS\datasets\nerf_synthetic\lego`
- 输出目录：**`outputs/phase_a_lego/`**（方法_数据集）
- 会在 step 101, 501, 1001, 3001, 5001 以及训练结束时保存 checkpoint

训练到 10000 步结束后再进行下面的验证。

---

## 2. 验证 Phase A（所有指定 step）

训练完成后，在**项目根目录**下执行（与训练一致，都用 experiments 下的脚本，仅参数不同）：

```bash
python experiments/run_validate.py --case lego_quaternion_phase_a --all_phase_a_steps
```

会对 step 101, 501, 1001, 3001, 5001, 10000 的 checkpoint 各跑一次 validate。验证结果写入 **`outputs/phase_a_lego/validate/step00101`**、**`step00501`** 等，与训练同属一实验目录。

若只想验证某一个 step，可以用：

```bash
python experiments/run_validate.py --case lego_quaternion_phase_a --step 10000
```

（将 `10000` 换成 101、501、1001、3001、5001 之一即可。）

---

## 3. 与 Baseline 对比

- **Baseline 训练**（与 Phase A 同数据、同步数，仅不用 quaternion renderer）：

  ```bash
  python experiments/run_fit.py --case lego_baseline
  ```

- **Baseline 验证**（例如只验证最终 step）：

  ```bash
  python experiments/run_validate.py --case lego_baseline --step 10000
  ```

对比同一 step 下 **`outputs/phase_a_lego/validate/stepXXXXX`** 与 **`outputs/baseline_lego/validate/stepXXXXX`** 中的 PSNR/SSIM/LPIPS 和渲染图即可。

---

## 4. 其它场景（chair）

`cases.py` 中已增加 `chair_quaternion_phase_a`，用法相同：

```bash
python experiments/run_fit.py --case chair_quaternion_phase_a
python experiments/run_phase_a_validate_all.py --case chair_quaternion_phase_a
```

---

## 5. 可选：快速冒烟测试

确认流程能跑通（仅 10 步）：

```bash
python experiments/run_fit.py --case lego_quaternion_phase_a --smoke
```

不会产生可用于对比的 checkpoint，仅用于检查配置与脚本是否正常。
