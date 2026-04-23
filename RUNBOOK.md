# Render package — 运行手册（tmux 测试 / 生产）

## 主入口（一条线跑全链路）

| 场景 | 命令 | 说明 |
|------|------|------|
| **推荐：看门狗闭环（全流程）** | `python pipeline_watchdog.py --config <yaml>` | 下载 → 渲染 → 编码 →（可选）上传；状态 `pipeline_state.json` |
| **只看门狗：只渲染** | `python pipeline_watchdog.py --config <yaml> --render-only` | 下载 → 仅 Blender；shard 全员 `mesh.ply` 后记 `render_only_done`，可按 `delete_source_shard_tar_after_render` 删 tar；**不编码、不上传** |
| **只看门狗：只编码** | `python pipeline_watchdog.py --config <yaml> --encode-only` | 不下载、不渲染；只对磁盘上已是 `render_done` 的 shard 跑 encode +（可选）上传；适合渲染跑完后第二阶段 |
| 单轮测试 | `python pipeline_watchdog.py --config <yaml> --once` | 只跑一轮后退出；可与 `--render-only` / `--encode-only` 组合 |
| 仅渲染+编码（本机已有 tar） | `SPCONV_ALGO=native python run_pipeline.py --config <yaml> --render_gpus 0,1 --encode_gpus 2,3` | 不下载、不看门狗；从 `paths.shard_dir` 取 `.tar.zst` |
| 仅补编码 | `SPCONV_ALGO=native python run_pipeline.py --config <yaml> --encode_only` | 扫 `render_dir` 未 encode 完成的 object |
| 仅下载 | `python download_trellis.py …` | 与 `hf.download` 参数对齐；见脚本 `--help` |
| 仅打包上传 | `python upload_hf_encoded_shards.py --config <yaml> --all-verified [--force]` | 仅 **严格** `encode_done` shard；Hub 已存在则跳过；`hf.upload.enabled: false` 时需 `--force`。校验 tar.zst / Hub 后按 YAML 删本地包与源 shard |
| 刷新上传台账 | `python refresh_upload_record.py --config <yaml>` | 按 `hf.upload` 列 Hub + 本地 `encode_done`（配置分片范围内），写入 `<data_root>/github/upload_record.json` |

**主入口 = `pipeline_watchdog.py`**（下载 + 渲染 + 编码 + 上传 + 状态 + 周期休眠）。

### 认证（Hugging Face）

- 环境变量（任选）：`HUGGINGFACE_HUB_TOKEN` 或 `HF_TOKEN`（优先级最高）。
- 或：在与主配置同目录下增加 **`<主配置名>.local.yaml`**（已加入 `.gitignore`），写入：
  ```yaml
  hf:
    upload:
      token: hf_...   # 勿提交、勿贴聊天
  ```
  `load_config` 会自动 deep-merge；`upload_hf_encoded_shards.py`、看门狗（上传开启时）、`hf_dataset_admin.py stats --config ...` 均可使用该 token。

---

## 环境与路径（测试前）

1. **进入仓库并激活编码环境**（与 Blender 渲染共用机器时，编码一般用 `envs/env`）：
   ```bash
   cd /mnt/hdd2/unified_model/render_package
   source envs/env/bin/activate   # 若未装见 environment_encode.yml / setup.sh
   ```
2. **权重与 Blender**：本地需有 `weights/`、`blender-3.5.1-linux-x64/`（未进 git，按 `pack.sh` / README 准备）。
3. **编辑配置**：`config/default.yaml` 或 `config/trellis_github_archives_6_first100.yaml`  
   - `paths.*`、`hf.download` 区间、`gpus.pool` / `gpus.render` / `gpus.encode`  
   - 上传：`hf.upload.enabled: true` 且 token（环境变量或 `.local.yaml`）；手动上传可在 `enabled: false` 时加 `--force`
4. **临时目录（避免满 `/tmp`）**：`load_config` 后若未设置 `TMPDIR`/`TEMP`/`TMP`，会默认指向 **`paths.render_tmp`**（数据盘）。Shell 若报 *here-doc / No space left on device*，可显式 `export TMPDIR=<data_root>/github/.render_tmp`。
5. **编码后端**：
   ```bash
   export SPCONV_ALGO=native
   ```

---

## 用 tmux 跑看门狗（推荐会话布局）

### 1）单会话 + 日志文件

```bash
cd /mnt/hdd2/unified_model/render_package
source envs/env/bin/activate
export SPCONV_ALGO=native
# Token：若用 .local.yaml 可省略下行
export HUGGINGFACE_HUB_TOKEN=   # 若开启 hf.upload 且未用 .local.yaml

LOGDIR="$HOME/logs/render_package"
mkdir -p "$LOGDIR"
SESSION="trellis_watchdog"

tmux new-session -d -s "$SESSION" -n run
tmux send-keys -t "$SESSION:run" \
  "cd $(pwd) && source envs/env/bin/activate && \
   export SPCONV_ALGO=native && \
   python pipeline_watchdog.py --config config/default.yaml 2>&1 | tee -a $LOGDIR/watchdog_\$(date +%Y%m%d).log" C-m

tmux attach -t "$SESSION"
```

- 断线：`Ctrl-b d`  
- 重连：`tmux attach -t trellis_watchdog`  
- 停跑：attach 后 `Ctrl-c`（看门狗会处理 SIGINT）

### 2）先单轮试跑（不长期循环）

```bash
tmux new-session -s trellis_test -n once
# 在 shell 里：
cd /mnt/hdd2/unified_model/render_package && source envs/env/bin/activate
export SPCONV_ALGO=native
python pipeline_watchdog.py --config config/default.yaml --once 2>&1 | tee ~/logs/watchdog_once.log
```

### 3）双窗格：看门狗 + nvidia-smi 监控（可选）

```bash
SESSION=trellis_watchdog
tmux new-session -d -s "$SESSION" -n main
tmux split-window -t "$SESSION:main" -h
tmux send-keys -t "$SESSION:main.0" "watch -n 5 nvidia-smi" C-m
tmux send-keys -t "$SESSION:main.1" \
  "cd /mnt/hdd2/unified_model/render_package && source envs/env/bin/activate && export SPCONV_ALGO=native && python pipeline_watchdog.py --config config/default.yaml" C-m
tmux attach -t "$SESSION"
```

---

## 状态与排错

| 产物 | 路径 / 含义 |
|------|----------------|
| 看门狗状态 | `<data_root>/github/pipeline_state.json`（**每整轮结束**写入；轮次进行中磁盘可能仍是上一轮快照） |
| 上传台账 | `<data_root>/github/upload_record.json`（`refresh_upload_record.py` 生成） |
| 渲染是否完成 | `render_dir/<shard>/<object>/mesh.ply` |
| 编码是否完成 | 见下：严格 vs 看门狗宽松 |
| Hub 是否已有包 | `python hf_dataset_admin.py stats <repo_id> --path-prefix github/render --repo-type dataset --config <yaml>` |

**Shard 状态（`run_pipeline._classify_shards`）**：标签是 **每个 shard 目录**，条件是该目录下 **每个物体子文件夹** 的与/或。

- **严格**（`upload_hf_encoded_shards`、不传 state 的 `_classify_shards`）：`encode_done` 当且仅当全员 `object_encode_complete`（各启用 stage 的 `latents/*`；`save_dino_features: false` 时可用下游 npz 推断）。
- **看门狗**（传入 `pipeline_state.json` 里的 `encode_passes`）：可选 `pipeline.watchdog.encode_retry_objects: never_started_only` —— 在某 shard **`encode_passes >= 1`** 后，只对「**仍无任何**对应 `latents/*.npz`**」的物体重试 encode；已有部分产物的物体视为「跑过」不再进队列。上传与台账统计仍用**严格** `encode_done`。

**进程说明**：`pgrep` 若见多个 `python3 pipeline_watchdog.py`，用 `pstree -p <主PID>` 确认：通常为 **1 个主进程 + 若干子进程**（子进程下挂 **Blender** = 渲染 worker），不是多实例看门狗。

常见问题：

- **Blender 找不到**：检查 `paths.blender_bin` 或仓库内 `blender-3.5.1-linux-x64/blender` 是否存在。  
- **上传失败**：token（env 或 `.local.yaml`）、`hf.upload.enabled` 或 `--force`、`repo_id`；远端已有同名包会跳过；体积不一致多为打包规则不同（如排除 `images`）。  
- **编码 OOM**：减小 `encode.batch_size` / `max_voxels`，或减少 `render.num_workers` / 并发 GPU 数；检查 `nvidia-smi` 是否有其它任务占满显存。

### 后台跑看门狗 + 日志落数据盘（示例）

```bash
cd /mnt/hdd2/unified_model/render_package
source envs/env/bin/activate
export SPCONV_ALGO=native
export TMPDIR=/path/to/<data_root>/github/.render_tmp
mkdir -p "$TMPDIR"
nohup python pipeline_watchdog.py --config config/trellis_github_archives_6_first100.yaml \
  >> /path/to/<data_root>/github/watchdog.log 2>&1 &
```

---

## 与 GitHub 仓库同步

```bash
cd /mnt/hdd2/unified_model/render_package
git pull
# 不在仓库内的目录：weights、envs、blender —— 按本地或 pack 流程准备
```

---

## 配置里与「测试」相关的项

- `pipeline.watchdog.interval_seconds`：每轮之间的休眠（秒）。  
- `pipeline.watchdog.shards_only_download: true`：只拉 `.tar.zst`，不拉 manifest（与当前默认一致）。  
- `pipeline.watchdog.encode_retry_objects`：`any_incomplete`（默认）或 `never_started_only`（见上文「Shard 状态」）。  
- `pipeline.delete_source_shard_tar`：上传/编码成功后是否删除 `shards/github/*.tar.zst`（省盘，慎用）。

试跑建议：复制一份 YAML（勿长期改坏默认文件），将 `hf.download.shard_index_start` 与 `shard_index_end` 设为同一索引（例如 `0`），并把 `pipeline.delete_source_shard_tar` 设为 `false`，再执行 `pipeline_watchdog.py --config <副本> --once`。确认下载 / 渲染 / 编码日志正常后，恢复全量区间、删包策略与 `--once`。
