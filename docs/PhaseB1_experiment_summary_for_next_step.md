# Phase B-1 实验总结（供下一步研究推进）

## 1. 实验目标

在 3D Gaussian Splatting 中引入 **Phase B-1**：用纯四元数表示外观颜色，并用一个全局可学习的四元数 rotor 做变换 `q_tilde = u q u_bar`，期望在保持与 baseline 可比的前提下验证该参数化与训练管线的正确性。

## 2. 实现概要

- **模型**：`QuaternionRotorGaussian` / `QuaternionRotorGaussianModel`  
  - 在 Vanilla Gaussian 基础上增加全局参数 `quat_rotor_raw` [4]，初始为 [1,0,0,0]（单位四元数，等价恒等变换）。  
  - 使用独立学习率 `quat_rotor_lr`（默认 0.001）训练该 rotor。
- **渲染器**：`QuaternionRotorRenderer`  
  - SH 转 RGB 后，将 RGB 视为纯四元数 `q = [0, r, g, b]`，用 `u`（对 `quat_rotor_raw` 归一化得到）做 `q_tilde = u q u_bar`，再取虚部得到最终 RGB 传入 rasterizer 的 `colors_precomp`。
- **Case**：`lego_quaternion_phase_b1`，数据集 Lego（Blender Synthetic），config 链：`blender.yaml` + `gsplat_v1.yaml` + `quaternion_phase_b1.yaml`。

## 3. 正常训练结果

- 训练与验证流程可完整跑通，得到 validate 指标与 viewer 结果。  
- 与 baseline（lego_baseline）相比：**指标与视觉效果基本一致**，未观察到明显提升或明显变差，说明当前 Phase B-1 的“正确实现”在默认设置下并未带来可见收益。

## 4. Broken Test 与结论

为确认 Phase B-1 的代码路径确实被使用（排除“没跑到该逻辑”的怀疑），做了 **broken test**：

- **做法**：将全局 rotor 固定为**非恒等的单位四元数**（绕 X 轴 90°：`[w,x,y,z]=[0.707, 0.707, 0, 0]`），且 **rotor 学习率设为 0**，使其在整个训练中不更新。
- **预期**：若 Phase B-1 的 rotor 路径被使用，颜色会被错误旋转，指标应明显差于 baseline；若未被使用，结果应与 baseline 一致。
- **实际**：Broken test 下验证指标明显变差（例如你提供的 PSNR 约 11–15、SSIM/LPIPS 等与正常 run 差异明显），说明**破坏 rotor 后模型确实失效**。
- **结论**：**Phase B-1 的 rotor 管线已被正确调用且对结果有影响**；当前“与 baseline 一致”的结果来自方法/超参本身，而非实现错误。代码层面可以认为验证通过，下一步应从方法或训练策略上寻求改进（如更合适的 rotor 初始化、正则、学习率或损失设计等）。

## 5. 技术细节（便于复现与扩展）

- **Rotor 初始化**：默认 [1,0,0,0]；支持可选 `init_rotor_broken_test`（用于上述 broken test，默认 false）。  
- **配置**：`configs/quaternion_phase_b1.yaml`；恢复为正常训练时使用 `quat_rotor_lr: 0.001`，且不开启 broken test。  
- **验证**：通过 `experiments/run_fit.py --case lego_quaternion_phase_b1` 训练，用现有 validate 与 `make_training_progress_figure.py` 等脚本评估。

## 6. 建议的下一步方向（供 ChatGPT/后续研究）

1. **分析与改进 rotor 的作用**：当前全局单一 rotor 可能表达能力有限，可尝试 per-Gaussian 或分层 rotor、或约束/正则使 rotor 偏离恒等时更稳定。  
2. **训练策略**：如 warmup、rotor 与其余参数的分阶段训练、或对 rotor 的梯度/范数做裁剪，观察是否改善收敛或指标。  
3. **损失与评估**：是否需要对颜色/rotor 增加辅助损失或评估指标，以更敏感地反映 Phase B-1 的贡献。  
4. **与 Phase A 的衔接**：若存在 Phase A（四元数解释/不同参数化），可明确 Phase B-1 在整体 pipeline 中的位置，以及是否与 Phase A 共享或扩展同一套四元数约定。

以上为 Phase B-1 的完整实验总结，可直接用于与 ChatGPT 的下一步研究推进讨论。
