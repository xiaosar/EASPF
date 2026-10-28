# EASPF

四个 SAR 处理程序，每个就是一个独立自足的 Python 文件，程序之间不互相 import。
每个模块目录里，程序与它的 `raw/` 样例输入、`results/` 输出放在一起。

## 程序

`01ST-GVD/stgvd.py`
子命令：`json-filter`、`par-filter`、`select-image`、`kml2shp`。
`json-filter` 按 AOI 重叠率筛归档 JSON，再用航向角、入射角与垂轨距离把影像归到同一条物理轨道，为每景输出一个 KML 足迹，并给每条轨道一份重叠率清单。
`par-filter` 在整个工作目录上做像对足迹重叠率扫描，据此改写 `itab` 的开关列。
`select-image` 为单景挑出可与之干涉的从影像。
`kml2shp` 把足迹合并成一个 Shapefile，可选择附带 overlap 属性。

`02RW-EKE/rweke.py`
子命令：`correct-orbit`、`delta-lonlat`。
`correct-orbit` 重构产品 XML 里的轨道：时域加权 RANSAC 选出共识集，加权最小二乘拟合三次多项式，观测覆盖区内用多项式内插、区外沿边界切线外推使拟合不发散，速度由多项式解析求导得到，最后按固定历元数回写。
`delta-lonlat` 配对同一景的 import/raw/correct 三类参数文件，报告中心坐标移动了多少。

`03GC-CPR/gccpr.py`
子命令：`register`、`stat-tif`、`stat-fft`。
`register` 在 GPU 上跑两遍：先用归一化互相关配合频域补零过采样得到整幅偏移场，再按偏移的整数部分把从影像窗口挪过去、只搜残差，最后用局部 Matrix-DFT 变焦细化到亚像素。
`stat-tif` 与 `stat-fft` 针对框出的区域，分别给出分段直方图与功率谱指标（高低频能量比、谱熵）。

`04FO-BEAM/fobeam.py`
没有子命令，直接跑一条九步流程：建立软链接、估计初始偏移、多视、生成差分干涉图、自适应滤波、解缠、去趋势、去大气、地理编码。
每一步都检查产物，已存在的自动跳过，因此可以断点续跑。

## 安装

```bash
python -m pip install -r requirements.txt
```

`stgvd.py`、`rweke.py` 需要 `numpy` + `shapely`；`gccpr.py register` 需要
`scipy` + `cupy`，它的两个统计子命令需要 `rasterio` + `matplotlib`；
`fobeam.py` 与 `stgvd.py` 的部分子命令调用外部干涉工具
（`create_offset` 至 `data2geotiff`、`mk_tab`、`base_calc`、`SLC_corners`、`ogr2ogr`）。

## 示例

```bash
unzip 01ST-GVD/raw/json.zip -d 01ST-GVD/raw      # 只需一次

python 01ST-GVD/stgvd.py json-filter \
    --aoi 01ST-GVD/results/MT-json/border.kml \
    --json-dir 01ST-GVD/raw/MT-json/json \
    --out-dir /tmp/out --min-overlap 0.05
```

所有子命令都有 `--help`，调用外部工具的那些还有 `--dry-run`。
