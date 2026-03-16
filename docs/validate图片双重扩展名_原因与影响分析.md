# Validate 输出图片出现 `.png.jpg` 双重扩展名：原因与影响分析

本文档仅做原因分析与影响说明，**不包含代码修改**，便于你决定是否在现有平台上改动。

---

## 一、现象

验证阶段保存的渲染图路径形如：

`outputs\phase_a_lego\validate\step00100\val\epoch=1-step=100\r_0.png.jpg`

文件名为 **`r_0.png.jpg`**，带有两个扩展名，不符合常规「单扩展名」习惯。

---

## 二、原因分析（数据流）

### 2.1 保存逻辑在哪里

**文件**：`internal/gaussian_splatting.py`  
**方法**：`save_images()`（约 584–619 行），在验证 epoch 结束时由工作线程把渲染结果写入磁盘。

构造路径的代码为：

```python
image_output_path = os.path.join(
    self.hparams["output_path"],
    item["stage"],
    "epoch={}-step={}".format(item["epoch"], item["step"]),
    "{}.jpg".format(item["image_name"].replace("/", "_"))
)
```

即：**最终文件名 = `item["image_name"].replace("/", "_")` + 固定后缀 `".jpg"`**。

### 2.2 `image_name` 从哪里来

- `item["image_name"]` 来自 **validation_step** 里入参 `image_info[0]`。
- `image_info` 由 **DataModule** 提供，其 `image_names` 来自各 **DataParser**。

对 **Blender / nerf_synthetic**（`internal/dataparsers/blender_dataparser.py` 约 41–43 行）：

```python
image_name_with_extension = "{}.png".format(frame["file_path"])
image_name_list.append(os.path.basename(image_name_with_extension))
```

- `frame["file_path"]` 在 nerf_synthetic 里一般是 `"r_0"` 这类无扩展名键。
- 因此 `image_name_with_extension` = `"r_0.png"`，再取 basename 得到 **`image_name = "r_0.png"`**。
- 即：**DataParser 给出的 `image_name` 已经带扩展名 `.png`**。

### 2.3 为何变成 `.png.jpg`

- 保存时**没有**去掉 `image_name` 原有扩展名，而是**直接**在后面再拼 `.jpg`：
  - `"{}.jpg".format(item["image_name"].replace("/", "_"))`
  - 等价于：`"r_0.png" + ".jpg"` → **`r_0.png.jpg`**。

所以：

- **直接原因**：保存逻辑始终在「完整 image_name」后追加 `.jpg`，没有先去掉已有扩展名。
- **数据来源**：Blender DataParser 提供的 `image_name` 已含 `.png`，其他 DataParser 若也带扩展名，会得到类似 `xxx.jpg.jpg` 等双重扩展名。

---

## 三、影响范围

### 3.1 已受影响的逻辑

| 位置 | 影响 |
|------|------|
| **验证阶段保存的图片路径** | 所有 validate 输出的渲染图都是 `原名.原扩展名.jpg`（如 `r_0.png.jpg`），与「单扩展名」惯例不一致。 |
| **experiments/make_training_progress_figure.py** | 通过 `IMAGE_NAME` 按**文件名**查找图片（`find_image_by_basename(eval_dir, image_name)`）。当前必须把 `IMAGE_NAME` 设为 `"r_0.png.jpg"`（或 `"r_1.png.jpg"` 等）才能找到现有输出，等于在配置上迁就了当前命名。 |

### 3.2 可能受影响的场景（若保持现状）

- **按扩展名解析类型的脚本**：若用 `Path.suffix` 等只取「最后一个」扩展名，多数会得到 `.jpg`，仍能当图片用；但若有人按「第一个」扩展名或「不含点号」的 basename 解析，可能误判或写错逻辑。
- **与原始数据集对齐**：数据集里是 `r_0.png`，验证结果是 `r_0.png.jpg`，名字不完全一致，做「同名对齐」时要额外处理。
- **文档/约定**：若文档或脚本约定「验证图与 dataset image 同名且仅改扩展名」，当前行为不符合该约定。
- **跨平台/工具**：少数工具或脚本可能假定「一个点一个扩展名」，双重扩展名可能带来兼容性或可读性问题（实际中很多地方仍能正常工作）。

### 3.3 不受影响或影响较小的部分

- **metrics CSV**：CSV 里记录的仍是 DataParser 的原始 `image_name`（如 `r_0.png`），不写磁盘文件名，因此不受双重扩展名影响。
- **PIL/torchvision 等读图**：通常按内容或最后一个扩展名识别格式，读 `r_0.png.jpg` 没问题。
- **Python `Path.suffix`**：对 `r_0.png.jpg` 得到 `.jpg`，仍可被 `list_image_files` 等按扩展名过滤的逻辑识别为图片。

---

## 四、修复思路（供你决定是否采用）

目标：**保存的验证图使用单一扩展名**，例如 `r_0.jpg`，且不改变现有目录结构或 stage/epoch/step 路径。

### 4.1 推荐做法（仅改保存时的文件名构造）

- **位置**：`internal/gaussian_splatting.py` 的 `save_images()` 中，构造 `image_output_path` 的那一段。
- **做法**：在拼 `.jpg` 之前，先把 `image_name` 的**已有扩展名去掉**，再统一加 `.jpg`：
  - 用 `os.path.splitext(item["image_name"])[0]` 或 `Path(item["image_name"]).stem` 得到无扩展名部分；
  - 再 `replace("/", "_")` 避免路径成分；
  - 最后拼 `+ ".jpg"`。
- **结果**：Blender 的 `r_0.png` → 存为 `r_0.jpg`；其他 parser 的 `xxx.jpg` → `xxx.jpg`（若先 strip 再拼 `.jpg` 则仍是 `xxx.jpg`），不会出现 `xxx.jpg.jpg`。

这样只改「写入时的文件名」，不改变保存目录、metrics、CSV 等其它逻辑。

### 4.2 若修复后需要同步调整的地方

- **experiments/make_training_progress_figure.py**：  
  - 当前 `IMAGE_NAME = "r_1.png.jpg"` 是为了匹配现有输出。  
  - 若平台侧改为输出 `r_0.jpg`，这里应改为 `IMAGE_NAME = "r_0.jpg"`（或对应视角的 `r_1.jpg` 等），否则会找不到文件。  
  - 若你暂时不改平台代码，只在自己脚本里用当前命名，则无需动 figure 脚本。

### 4.3 不修改时的注意点

- 保持现状也可以正常运行，只是：
  - 需在 `make_training_progress_figure.py` 等处用 `r_0.png.jpg` 这类名字查找；
  - 文档或对外说明里最好注明「验证图文件名为 原名.原扩展名.jpg」，避免他人误以为是错误。

---

## 五、总结

| 项目 | 说明 |
|------|------|
| **原因** | DataParser（如 Blender）给出的 `image_name` 已带 `.png`；保存时又直接在其后拼接 `.jpg`，未先去掉原扩展名，得到 `r_0.png.jpg`。 |
| **影响** | 主要是命名不常规、以及依赖「完整文件名」的脚本（如 make_training_progress_figure）必须写 `r_0.png.jpg`；metrics、读图等多数不受影响。 |
| **修复** | 在 `gaussian_splatting.save_images()` 里用「无扩展名 basename + `.jpg`」构造文件名即可；若修复，figure 脚本中的 `IMAGE_NAME` 需改为单扩展名。 |
| **是否修改** | 由你根据是否在意命名规范、是否希望与后续脚本/文档一致来决定；不修改也能继续用，修改则改动小、风险可控。 |

以上仅为分析与方案说明，未对任何代码做修改。
