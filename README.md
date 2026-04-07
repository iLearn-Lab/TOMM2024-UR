# Universal Relocalizer for Weakly Supervised Referring Expression Grounding

> A plug-and-play relocalization framework for weakly supervised referring expression grounding that improves object localization by refining proposal scores with category, color, and spatial cues.

## Authors

PANPAN ZHANG<sup>1</sup>, MENG LIU<sup>2*</sup>, XUEMENG SONG<sup>3*</sup>, DA CAO<sup>4</sup>, ZAN GAO<sup>5</sup>, LIQIANG NIE<sup>6</sup>

<sup>1</sup> Shandong University, Qingdao, China  
<sup>2</sup> Shandong Jianzhu University, Jinan, China   
<sup>3</sup> Southern University of Science and Technology, Shenzhen, China    
<sup>4</sup> Hunan University, Changsha, China  
<sup>5</sup> Qilu University of Technology, Jinan, China  
<sup>6</sup> School of Computer Science and Technology, Harbin Institute of Technology (Shenzhen), Shenzhen, China  

<sup>*</sup> Corresponding author


## Links

- **Paper**: [`Paper Link`](<[paper-link](https://dl.acm.org/doi/10.1145/3656045)>)
- **Project Page**: [`Project Page`](<project-page-link>)
- **Hugging Face Model**: [`Model`](<huggingface-model-link>)
- **Hugging Face Dataset**: [`Dataset`](<huggingface-dataset-link>)
- **Demo / Video**: [`Demo`](<demo-link>)
- **Code Repository**: [`GitHub`](https://github.com/iLearn-Lab/<repo-name>)


---

## Table of Contents

- [Updates](#updates)
- [Introduction](#introduction)
- [Highlights](#highlights)
- [Method / Framework](#method--framework)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Checkpoints / Models](#checkpoints--models)
- [Dataset / Benchmark](#dataset--benchmark)
- [Usage](#usage)
- [Demo / Visualization](#demo--visualization)
- [TODO](#todo)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)
- [License](#license)

---

## Updates

- [04/2026] Initial release




---

## Introduction



We present **Universal Relocalizer**, a plug-and-play framework for **weakly supervised referring expression grounding**. It improves existing two-stage methods by refining proposal scores with category, color, and spatial relationship cues, leading to more accurate object localization without requiring region-level annotations. This repository provides the official implementation, along with training and evaluation code on standard benchmarks.

---

## Highlights

- 支持 `<task / domain>`
- 提供 `<training / inference / evaluation>` 脚本
- 提供 `<checkpoint / dataset / benchmark / demo>`
- 适合用于 `<论文复现 / 项目展示 / 后续研究>`

---

## Framework

你可以在这里放方法框架图、模型结构图或整体 pipeline 图。

### Framework Figure

```markdown
![Framework](./assets/framework.png)
```

实际使用时，把上面这行替换成：

```markdown
![Framework](./d.png)
```

然后在下面补一句说明：

**Figure 1.** Overall framework of `<UR>`.

---

## Project Structure

```text
.
├── assets/                # 图片、框架图、结果图、demo 图
├── configs/               # 配置文件
├── data/                  # 数据说明（不建议直接上传大数据本体）
├── scripts/               # 训练、推理、评测脚本
├── src/                   # 核心源码
├── README.md
├── requirements.txt
└── LICENSE
```

如果你的项目结构不同，请按实际情况修改。

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/iLearn-Lab/<repo-name>.git
cd <repo-name>
```

### 2. Create environment

```bash
python -m venv .venv
source .venv/bin/activate   # Linux / Mac
# .venv\Scripts\activate    # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> 如果你使用的是 conda、poetry、uv 或 docker，请改成自己的实际安装方式。

---

## Checkpoints / Models

如果你们发布了模型权重，可以写：

- **Main checkpoint**: [`Model Link`](<huggingface-model-link>)
- **Additional checkpoint**: [`Other Checkpoint`](<other-checkpoint-link>)

下载后请放入如下目录：

```text
checkpoints/
```

如果需要修改配置路径，也可以说明：

- 修改 `config.yaml` 中的 checkpoint 路径
- 或在运行脚本时通过参数传入

---

## Dataset / Benchmark

如果你们还提供数据集，可以写：

- **Dataset**: [`Dataset Link`](<huggingface-dataset-link>)
- **Benchmark**: [`Benchmark Link`](<benchmark-link>)

并说明数据组织方式，例如：

```text
data/
├── train/
├── val/
└── test/
```

> 如果数据集不能直接公开，请在这里说明申请方式或访问限制。

---

## Usage

### Training

```bash
python scripts/train.py
```

### Inference

```bash
python scripts/infer.py
```

### Evaluation

```bash
python scripts/eval.py
```

请根据你的项目实际情况替换成真实命令。  
如果你的项目没有训练或评测部分，可以删除对应小节。

---

## Demo / Visualization

如果你们有演示页面、视频或截图，可以写在这里。

### Demo Video

- [`Demo Link`](<demo-link>)

### Example Results

你可以插入结果图：

```markdown
![Result](./assets/result.png)
```

或者放一个简单结果表：

| Setting | Result |
|---|---:|
| Baseline | xx.x |
| Ours | xx.x |

---

## TODO

- [ ] 完善文档
- [ ] 补充训练脚本说明
- [ ] 补充推理脚本说明
- [ ] 上传模型权重
- [ ] 上传结果图
- [ ] 发布 demo / project page

---

## Citation

```bibtex
@article{zhang2024universal,
  title={Universal relocalizer for weakly supervised referring expression grounding},
  author={Zhang, Panpan and Liu, Meng and Song, Xuemeng and Cao, Da and Gao, Zan and Nie, Liqiang},
  journal={ACM Transactions on Multimedia Computing, Communications and Applications},
  volume={20},
  number={7},
  pages={1--23},
  year={2024},
  publisher={ACM New York, NY}
}
```

---

## Acknowledgement


- This project benefits from the open-source implementation of [DTWREG](https://github.com/insomnia94/DTWREG).  
We sincerely thank the authors for making their code publicly available.

---

## License

This project is released under the Apache License 2.0.
