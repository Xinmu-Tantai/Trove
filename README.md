# Trove

一个用于沉淀论文与技术文章原始内容的 Markdown 知识库。每篇笔记按统一模板整理公开资源、摘要和引言的中英对照、作者提出的贡献、方法、数据集、实验、局限与重要参考文献。

本仓库只记录可追溯的来源内容及其忠实翻译，不加入个人评价、推测或延伸解读。

## 目录结构

```text
Trove/
├── articles/                  # 已完成的文章笔记
│   ├── <paper-title>.md       # 文章笔记
│   └── images/                # 从原文裁剪的图表等附件
├── sources/                   # 待整理的原文 PDF、网页导出等
├── templates/
│   └── article-template.md    # 文章笔记模板
├── work/                      # 临时脚本、页面渲染与中间文件
└── AGENTS.md                  # 仓库工作约定
```

`sources/`、`work/` 与 `articles/` 会在添加文章时创建；临时产物应留在 `work/`，不要混入正式笔记。

## 新增一篇文章

1. 将原文放入 `sources/`。
2. 为文章选择稳定、简短的标识，例如 `attention-is-all-you-need`。
3. 复制 [文章模板](templates/article-template.md)，以论文标题保存为 `articles/<paper-title>.md`。
4. 按模板提取并填写内容；原文图表裁剪后放入 `articles/images/`，并以文章标识作为文件名前缀，再用相对路径引用。
5. 提交前检查来源链接、图表路径、数字、公式与引用编号，并删除未填写的占位符。

## 笔记规范

- 优先依据文章全文及其附录；官方项目主页、代码仓库和文档只作为补充来源，并标明链接。
- 摘要与引言保留完整原文，并提供逐段对应的中文翻译。
- 贡献、方法和实验仅整理作者明确说明的内容；作者观察与作者结论应清楚区分。
- 未获取或原文未说明的信息应明确标记，不能推断补全。
- 使用 Markdown 相对路径，确保笔记与附件可随仓库一起浏览。

## 准备论文原文

可用脚本完成下载、逐页文本提取和页面渲染等重复步骤。例如：

```bash
python3 scripts/prepare_paper.py \
  --url https://arxiv.org/pdf/2406.09246 \
  --article-id openvla \
  --title "OpenVLA: An Open-Source Vision-Language-Action Model" \
  --render-pages 1,4,7,9,10
```

脚本以论文标题命名 PDF 并保存到 `sources/`，把逐页文本、页面渲染和清单写入 `work/<article-id>/`。为跨平台兼容，文件名中的 `:`、`/` 等保留字符会替换为 `-`。页面渲染需要安装 PyMuPDF：

```bash
python3 -m pip install PyMuPDF
```

需引用原文图表时，可用 [裁剪脚本](scripts/crop_pdf_region.py) 从指定 PDF 页面导出图像。例如：

```bash
python3 scripts/crop_pdf_region.py \
  --pdf "sources/Paper Title.pdf" --page 4 \
  --rect 100,60,520,290 \
  --output articles/images/<article-id>-figure-2.png
```

## 模板

笔记模板见 [templates/article-template.md](templates/article-template.md)。其内容包括：

- 基本信息与开源资源
- 摘要、引言的原文和中文翻译
- 创新点、核心方法、数据集与框架
- 实验、局限与结论
- 重要参考文献

## 许可

除非各文章笔记另有说明，原始论文、图表、代码及其他外部材料仍适用其各自的版权和许可。
