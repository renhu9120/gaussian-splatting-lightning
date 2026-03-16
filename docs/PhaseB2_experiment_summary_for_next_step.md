# Phase B-2 实验总结（供下一步研究推进）

## 1. 实验目标

在 Phase B-1（全局单一 quaternion rotor）基础上，推进 **Phase B-2**：将“全局 rotor”升级为**每个 Gaussian 一个 rotor**，即：

- 公式：`q_tilde_i = u_i q_i u_i_bar`
- 其中 `q_i` 为第 i 个 Gaussian 的 pure quaternion 颜色，`u_i` 为该 Gaussian 的可学习单位四元数 rotor。

目标：在保持 pure quaternion 框架与最小工程改动的前提下，引入**局部自适应的四元数通道耦合**，观察是否相对 B-1 / baseline 出现可辨别的收益。

## 2. 实现概要

- **模型**：`PerGaussianRotorGaussian` / `PerGaussianRotorGaussianModel`
  - 在 Vanilla Gaussian 基础上增加 per-Gaussian 属性 `quat_rotors_raw` [N, 4]，每行初始为 [1, 0, 0, 0]（恒等）。
  - 使用独立学习率 `quat_rotors_lr = 0.0003` 训练 rotors（较 B-1 略小以稳定 per-Gaussian 参数量）。
- **渲染器**：`PerGaussianRotorRenderer`
  - SH → RGB 后，将每个 Gaussian 的 RGB 视为 pure quaternion `q_i`，用对应的归一化 `u_i` 做 `q_tilde_i = u_i q_i u_i_bar`，取虚部得到 RGB，再以 `colors_precomp` 送入 rasterizer。
- **Case**：`lego_quaternion_phase_b2`，数据集 Lego（Blender Synthetic），config 链：`blender.yaml` + `gsplat_v1.yaml` + `quaternion_phase_b2.yaml`。
- **训练**：max_steps=10000，save_iterations=(101, 501, 1001, 3001, 5001)，与 baseline / phase_a / phase_b1 对齐。

## 3. 验证结果（Phase B-2 完整跑 10k steps）

程序运行正常，validate 与 progress figure 均成功生成。指标汇总如下（Lego 验证集）：

| step | loss    | rgb_diff | ssim   | psnr   | lpips  |
|------|---------|----------|--------|--------|--------|
| 100  | 0.1229  | 0.0642   | 0.642  | 17.79  | 0.376  |
| 500  | 0.0596  | 0.0292   | 0.819  | 22.47  | 0.249  |
| 1000 | 0.0440  | 0.0202   | 0.861  | 24.69  | 0.159  |
| 3000 | 0.1480  | 0.1157   | 0.723  | 12.34  | 0.280  |
| 5000 | 0.0112  | 0.0068   | 0.971  | 32.95  | 0.026  |
| 10000| 0.0078  | 0.0052   | 0.982  | 35.33  | 0.014  |

- 训练稳定，无 NaN/崩溃；最终 PSNR ≈ 35.3，SSIM ≈ 0.98，LPIPS ≈ 0.014，与典型 Lego baseline / B-1 同量级。
- **与 baseline / B-1 相比：未观察到明显、一致的提升**；曲线形态（如 3k 附近波动后收敛）与既有实验类似，整体仍处于“与 baseline 相当”的水平。

## 4. 结论与待解释点

- **实现与稳定性**：Phase B-2 的 per-Gaussian rotor 管线已正确接入并稳定训练至 10k steps；densification 等流程兼容 `quat_rotors_raw`，无异常。
- **收益**：在当前设置（同一数据、同一训练协议、未改 loss/正则/几何）下，**per-Gaussian 四元数耦合并未带来可辨别的指标优势**，与 Phase B-1 的结论一致——四元数 rotor 机制本身工作正常，但**单纯增加“局部 rotor”表达能力并未在现有目标下体现为更好重建质量**。
- **可能原因（供讨论）**：
  1. 重建 loss（rgb_diff + SSIM）对“颜色通道的四元数旋转”不敏感，rotor 学到的变化对指标贡献有限；
  2. 无额外正则或约束时，rotor 可能倾向于保持接近恒等，实际耦合较弱；
  3. 需要 view-dependent、更强先验（如 luminance/chroma 解耦）或不同损失/评估方式，才能体现四元数建模的差异；
  4. 与 Phase A 的“四元数解释”尚未形成联合设计，当前 B-2 仍是“SH → RGB → 再经 rotor”的独立模块。

## 5. 技术细节（便于复现与扩展）

- **配置**：`configs/quaternion_phase_b2.yaml`；`quat_rotors_lr: 0.0003`。
- **训练**：`python experiments/run_fit.py --case lego_quaternion_phase_b2`。
- **验证与作图**：沿用现有 `run_validate` 与 `make_training_progress_figure.py`（experiment 设为 `phase_b2_lego`）。

## 6. 建议与 ChatGPT 讨论的下一步方向

1. **是否引入 view-dependent rotor**：rotor 是否应依赖视角（如用 SH 或 MLP 生成 u_i(dir)），以更好捕捉观察依赖的颜色耦合。
2. **损失与正则**：是否需要对 rotor 偏离恒等施加温和正则、或增加与颜色/外观相关的辅助损失，使四元数耦合对训练信号更敏感。
3. **评估与诊断**：是否增加针对“颜色耦合程度”的简单诊断（如 mean(|rgb_rot - rgb|)、rotor 偏离恒等的统计），以判断 rotor 是否在学习非平凡变换。
4. **与 Phase A 的整合**：Phase A 若已有四元数解释/参数化，B-2 的 per-Gaussian rotor 应如何与之衔接（共享约定、共享表示或分阶段训练）。
5. **更激进的方向**：是否在本阶段考虑 quaternion decoder、luminance/chroma 分离、第四通道或 Dirac 正则等（此前 Phase B-2 明确未做），或先保持最小改动、仅在上面的 1–4 中选一到两个做对照实验。

---

以上为 Phase B-2 的完整实验总结，可直接复制给 ChatGPT 用于下一步研究推进与方案讨论。
