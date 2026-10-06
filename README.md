# Camera Calibration

独立的 xArm6 + RealSense 相机外参标定工具，支持 ChArUco 与 ArUco GridBoard。你用外部遥操移动机械臂，本工具读取状态、显示标定板、按空格保存停稳后的数据，最后输出 JSON。

**代码、注释、终端和窗口提示均为英文。** 本 README 用中文介绍操作。

## 范围

- 固定相机：板固定在机械臂末端，求相机彩色光学坐标到机械臂基座的变换。
- 腕部相机：板固定在桌面，求相机彩色光学坐标到 TCP 的变换，并保留到法兰的变换。
- 读取控制器当前 payload、质心、TCP offset、world offset，不使用 ASPIRE 或遥操项目的配置。
- 不发送移动、回家、开闭夹爪、使能、清错或设置参数指令。退出只关闭相机和连接，不控制机械臂停止。
- 板类型由 JSON 的 `board_type` 显式选择，不根据 dictionary 自动猜测。旧配置未写类型时按 ChArUco 处理。
- 首版支持现代 xArm6 rich 状态报文，SDK 固定为 1.18.4；屏蔽该 SDK 连接握手里自动清除告警的行为。
- 只读取彩色流，默认 640×480 / 30 FPS，使用该流设备内参，不重新标定镜头内参或深度。
- 所有输出路径必须位于本仓库内，支持离线复算，不自动更新其他项目。

## 环境

推荐在机械臂电脑的 Linux 桌面终端运行。需要 Python 3.11、USB 相机访问权限、本机显示环境及可连接的机械臂 IP。

本工作区已建立 `.venv`。重新安装时在此目录执行；`/path/to/python3.11` 替换为已有解释器路径：

```bash
cd /home/siqis3/workspace/camera_calibration
mkdir -p .cache
export UV_CACHE_DIR="$PWD/.cache/uv"
export TMPDIR="$PWD/.cache"
export XDG_CACHE_HOME="$PWD/.cache"
export UV_PYTHON_DOWNLOADS=never
uv venv --python /path/to/python3.11 .venv
uv pip install --python .venv/bin/python -e '.[test]'
```

使用带本机窗口的 `opencv-contrib-python`，不要同时安装 `opencv-python` 或 headless 版本。所有依赖安装在本仓库 `.venv`，不修改系统环境或其他项目。

## 1. 查看相机

```bash
.venv/bin/camera-calibration devices
```

输出型号和序列号。根据序列号选择，不依赖 USB 枚举顺序。采集时只能由一个程序占用该相机；请关闭占用它的 RealSense Viewer 或其他录像程序。外部遥操只负责机械臂移动。

### 只预览画面、调整相机位置

```bash
.venv/bin/camera-calibration preview
```

按提示选择相机；也可以加 `--serial 311322303806` 直接打开指定相机。窗口可缩放，按 **Q / Esc** 或关闭窗口退出。这个入口不连接机械臂、不需要标定板、不保存图像或标定数据。固定相机调整安装位置后，应重新标定。

## 2. 准备标定

1. 在 Studio 中完成实际负载和 TCP 设置。本工具会读取这些设置，不替你修改。第一版要求 world offset 为零，非零时会显示值并退出。
2. 固定相机保持不动，标定板牢固固定在末端；腕部相机保持与末端刚性连接，标定板固定在桌面。
3. 本轮采集中不得改变板、相机或机械臂基座的安装。不需要提前测量板相对末端或桌面的位置。
4. 默认 ChArUco 板为 4×4 格、格边 30 mm、标记黑色外框边长 22 mm。使用兼容的 `DICT_4X4_50`；程序检查 ID 布局，区分现代/legacy 排列，无法确定则拒绝采样。这不意味着唯一识别了原打印字典容量。
5. 启动遥操可能回家或开夹爪，务必在采集前处理好，防止固定板松动。采样前停止发送运动指令。

### 标定板配置

通过 `--board 文件.json` 选择板配置。**两个示例中的尺寸均需按实物核实，GridBoard 的 60 mm / 5 mm 仅为示例，不是附图实测结果。**

| 字段 | ChArUco | GridBoard |
| --- | --- | --- |
| `board_type` | `charuco` | `gridboard` |
| 行列数 | `squares_x`、`squares_y`：棋盘格数量 | `markers_x`、`markers_y`：标记数量 |
| `square_length_m` | 一个完整棋盘格的边长 | 不使用 |
| `marker_length_m` | 标记黑色外框边长 | 标记黑色外框边长 |
| `marker_separation_m` | 不使用 | 相邻黑框之间的完整白色间隙 |
| `dictionary` | 标记编码字典 | 标记编码字典 |
| `legacy_pattern` | `null` 自动判断，或 `true` / `false` | 不使用 |

所有长度以米为单位；标记边长包含黑色边框、不含外部白边。GridBoard 不填写 `square_length_m`，也不需要计算“标记边长加白边”。`DICT_4X4_50` 中的 `4X4` 表示编码位数，与板行列数无关。

- `board.charuco.example.json`：显式 ChArUco 示例，4×4 格、30 mm 格长、22 mm 标记。
- `board.gridboard.example.json`：显式 GridBoard 示例，3×3 标记、60 mm 标记、5 mm 间隙。
- `board.example.json`：保留原示例路径，内容仍是 ChArUco。

第一版 GridBoard 使用 OpenCV 标准 ID 布局：从板自身左上角的 ID 0 开始，逐行连续编号，标记朝向按标准打印图。旋转整块板不会改变配置；不支持任意重新排列或单独旋转标记。棋盘坐标 X 沿列方向、Y 沿行方向；不能因为画面旋转就交换参数。

若 OpenCV 提示标记间隙偏窄，这是打印布局提示；仍须填写真实间隙，不能改大配置值来消除提示。

启动时先验证配置并打印类型与尺寸，再连接设备。错误类型、混用字段、缺少字段、非法尺寸或字典会给出英文提示。

未写 `board_type` 的旧配置和旧会话会在内存中按 ChArUco 解释，续采不会重写旧 `session.json`。用户原有的 `board_fixed_tcp.json` 和 `board_wrist_table.json` 无需更改。

## 3. 开始采集

最简单的交互方式：

```bash
.venv/bin/camera-calibration capture
```

按提示输入机械臂 IP、相机序号、`fixed` 或 `wrist`。也可以指定参数：

```bash
.venv/bin/camera-calibration capture \
  --robot-ip 192.168.1.209 \
  --mount fixed \
  --name fixed_main \
  --session sessions/fixed_main_01
```

使用 GridBoard 时，先将示例中的尺寸改成实测值，然后单独建立会话：

```bash
.venv/bin/camera-calibration capture \
  --robot-ip 192.168.1.209 \
  --serial 215122252880 \
  --mount wrist \
  --name xarm_wrist \
  --board board.gridboard.example.json \
  --session sessions/wrist_gridboard_01
```

这里的 IP 只是示例；使用当前控制器实际地址。添加 `--serial SERIAL` 可以直接选择相机。腕部相机使用 `--mount wrist --name wrist`，并单独创建会话目录。

窗口操作：

- **空格**：停稳后采样，程序检查图像与前后位姿，再保存。
- **Q / Esc / 关闭窗口**：结束本次采集，保留已保存的数据。
- 窗口显示板类型、标记、用于求解的角点数量、角点误差、已保存数量和最近一次结果；较长的拒绝原因完整显示在终端。

先采 15 组，覆盖不同位置和不同轴向的倾斜。不是只做 XYZ 平移，也不需要靠近机械臂工作空间边界。让板占据足够大的画面面积、清楚入镜。

采集检查：

控制器状态 `5`（`MODE_CHANGED`）常在修改 mode、payload 或 TCP offset 后出现，不等同于错误。本工具允许在该状态下读取并采样，仍要求参数一致、位姿静止、状态报文新鲜且无错误/告警；不会调用 `set_state(0)` 或使能机械臂。需要移动时由外部遥操或 Studio 管理运动状态。

- 前后共 8 个新鲜机械臂状态，跨度约 0.6 秒以上；位置变化不超过 0.2 mm、旋转变化不超过 0.002 rad。
- ChArUco 至少 6 个非共线棋盘角点；GridBoard 至少 3 个有效标记且标记中心不能共线，使用每个标记的四个外角点进行亚像素细化和求解。两者都检查布局一致性、正深度和位姿歧义。
- 与任意已有样本同时小于 10 mm 位移和 5° 旋转时，认为重复。
- 参数改变会锁定本轮采样，退出并建立新会话；通信错误、运动或视觉质量不够时拒绝当前样本。
- 相机持续取流，跳过重复帧；采样时丢弃缓冲帧。此流程基于静止前后配对，**不是硬件同步**。

### 续采

```bash
.venv/bin/camera-calibration capture --session sessions/fixed_main_01
```

沿用原会话的设备与板配置，重新读取设备参数核对，并要求确认板和相机安装没有变化。新增样本不会覆盖旧样本；已有 `result.json` 会标为需重新求解。安装变动、换板、换类型、改尺寸或改字典后必须新建会话，不能仅修改旧文件继续采集。

## 4. 离线求解

```bash
.venv/bin/camera-calibration solve --session sessions/fixed_main_01
```

不需要机械臂、相机或显示器，也不需要重新指定板类型。程序按会话配置重新读取原始图像并检测，不把保存过的检测结果当成最终答案。

- 有效样本至少 12 组，旋转必须覆盖不平行的轴。
- 使用 Park 手眼算法，先逐组留出验证，再使用全部有效样本输出结果。
- 默认单组角点 RMS ≤ 1.5 像素；这是归一化光学坐标乘焦距后的**去畸变图像平面等效像素误差**。
- 默认留出平移 RMS ≤ 5 mm、旋转 RMS ≤ 2°。这些是初始数值验收标准，不保证实际抓取误差。
- 留出误差较大时查看 `leave_one_out` 各组结果，检查安装、尺寸、成像质量和姿态覆盖后补采或重采；不为达到通过而自动删掉高误差组。
- 图像检测或完整性检查失败的组写入 `excluded_samples`，不会默默计入有效样本。

可选阈值：

```bash
.venv/bin/camera-calibration solve --session sessions/fixed_main_01 \
  --max-corner-rms-px 1.5 --max-translation-mm 5 --max-rotation-deg 2
```

验证通过返回码为 0；未通过为 2。未通过也保存结果及原因，外参可能为空或为未通过验证的候选值。下游使用前应检查 `validation.passed`。

## 5. 输出文件

```text
sessions/fixed_main_01/
  session.json
  sample_001/
    color.png
    sample.json
  sample_002/
    ...
  rejections.jsonl
  result.json
```

`rejections.jsonl` 在发生拒绝时创建。原始图像、当时的参数和完整配对位姿保留在会话目录，不混入此前 ASPIRE 的数据。重新 solve 会更新当前会话的 `result.json`。

`result.json` 固定为 JSON，不含 `schema_version`。主要字段：

| 字段 | 含义 |
| --- | --- |
| `camera` | 型号、唯一序列号和用户名称 |
| `mount_type` | `fixed` 或 `wrist` |
| `intrinsics` | 彩色流分辨率、内参矩阵、设备原始畸变模型与系数 |
| `extrinsics` | 源/目标坐标系、4×4 矩阵；腕部另有 `flange` 结果 |
| `robot_settings` | payload、法兰坐标下质心、TCP offset、world offset |
| `calibration` | 时间、含显式 `board_type` 的完整板规格、Park 方法、使用及排除样本 |
| `validation` | 是否通过、原因、阈值和逐组留出误差 |
| `units` | 米、弧度、千克 |

矩阵用于齐次列向量：`p_target = T_target_source @ p_source`。

- 固定：`color_optical_frame → robot_base`。
- 腕部：`color_optical_frame → robot_tcp`，另存 `color_optical_frame → robot_flange`。
- 光学轴约定：X 向右、Y 向下、Z 向前。矩阵按行嵌套存储，平移在最后一列。
- 内参只适用于记录的流和分辨率。彩色外参不能直接当作原始深度光学坐标系的外参。
- TCP offset 表示法兰到 TCP 的固定变换，RPY 使用 `Rz(yaw) @ Ry(pitch) @ Rx(roll)`。
- 腕部相机安装不变但 TCP offset 更新时，用保存的法兰外参换算：`T_tcp_camera = inverse(T_flange_tcp) @ T_flange_camera`。

## 测试与已知限制

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q
```

测试只使用合成图像、已知变换和模拟设备，临时测试数据在本仓库 `.cache/tests`。覆盖固定/腕部求解、非零 TCP 偏移、ChArUco 新旧布局、GridBoard 遮挡与布局检查、旧会话兼容、畸变处理、静止/重复检查、报文读取、断连、续采和 JSON 导出。

RealSense `none`、`brown_conrady`、`inverse_brown_conrady` 通过 SDK 反投影为归一化光学坐标；其他畸变模型明确拒绝，不静默当成零畸变。镜头内参/纸板尺寸的系统误差、机械臂绝对精度和安装挠曲仍可能影响实际结果。

无桌面显示环境时 `capture` 会提前退出，避免连接机械臂后才发现不能显示窗口。启动成功不代表标定已完成；应由操作者进行新一轮真实采集，并检查数值结果和实际空间对应关系。
