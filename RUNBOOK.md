# Render package — 运行手册（tmux 测试 / 生产）

## 主入口（一条线跑全链路）

| 场景 | 命令 | 说明 |
|------|------|------|
| **推荐：看门狗闭环** | `python pipeline_watchdog.py --config <yaml>` | 校验配置 → 按区间下载 → 解压/渲染 → 编码 →（可选）打包上传；循环；状态写入 `data_root/github/pipeline_state.json` |
| 单轮测试 | `python pipeline_watchdog.py --config <yaml> --once` | 只跑一轮后退出，适合 cron / 手工试跑 |
| 仅渲染+编码（本机已有 tar） | `SPCONV_ALGO=native python run_pipeline.py --config <yaml> --render_gpus 0,1 --encode_gpus 2,3` | 不下载、不看门狗；从 `paths.shard_dir` 取 `.tar.zst` |
| 仅补编码 | `SPCONV_ALGO=native python run_pipeline.py --config <yaml> --encode_only` | 扫 `render_dir` 未 encode 完成的 object |
| 仅下载 | `python download_trellis.py …` | 与 `hf.download` 参数对齐；见脚本 `--help` |
| 仅打包上传 | `python upload_hf_encoded_shards.py --config <yaml> --all-verified` | 仅 `encode_done` shard；校验 tar.zst / Hub 后删本地包与源 shard（见配置） |

**主入口 = `pipeline_watchdog.py`**（下载 + 渲染 + 编码 + 上传 + 状态 + 周期休眠）。

---

## 环境与路径（测试前）

1. **进入仓库并激活编码环境**（与 Blender 渲染共用机器时，编码一般用 `envs/env`）：
   ```bash
   cd /mnt/hdd2/unified_model/render_package
   source envs/env/bin/activate   # 若未装见 environment_encode.yml / setup.sh
   ```
2. **权重与 Blender**：本地需有 `weights/`、`blender-3.5.1-linux-x64/`（未进 git，按 `pack.sh` / README 准备）。
3. **编辑配置**：`config/default.yaml` 或 `config/trellis_github_archives_6_first100.yaml`  
   - `paths.*`、`hf.download` 区间、`gpus.render` / `gpus.encode`（建议互不重叠）  
   - 上传：`hf.upload.enabled: true` 且 `export HUGGINGFACE_HUB_TOKEN=...`
4. **编码后端**：
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
export HUGGINGFACE_HUB_TOKEN=   # 若开启 hf.upload

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
| 看门狗状态 | `<data_root>/github/pipeline_state.json` |
| 渲染是否完成 | `render_dir/<shard>/<object>/mesh.ply` |
| 编码是否完成 | `run_pipeline` 的 `_classify_shards`：`latents/` 下各 stage 产物 |
| Hub 是否已有包 | `hf_dataset_admin.py stats <repo_id> --path-prefix github/render --repo-type dataset` |

**与看门狗一致的进度判定**：`run_pipeline._classify_shards` 按 shard 目录聚合；`encode_done` 要求该 shard 下**每个**物体都具备启用 stage 的 `latents/*` 产物（`save_dino_features: false` 时可用 `unilat`/`slat`/`ss` 代替已删除的 `dino_features.npz`）。`render_done` 表示全员已有 `mesh.ply` 但至少有一个物体未 encode 完。可在本机用 `load_config` → `pipeline_watchdog.expected_stems` → `_classify_shards` 做统计，与看门狗打印的 `status` 对齐。

常见问题：

- **Blender 找不到**：检查 `paths.blender_bin` 或仓库内 `blender-3.5.1-linux-x64/blender` 是否存在。  
- **上传失败**：`HUGGINGFACE_HUB_TOKEN`、`hf.upload.enabled`、`repo_id`。  
- **编码 OOM**：减小 `encode.batch_size` / `max_voxels`，或减少 `render.num_workers`。

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
- `pipeline.delete_source_shard_tar`：上传/编码成功后是否删除 `shards/github/*.tar.zst`（省盘，慎用）。

试跑建议：复制一份 YAML（勿长期改坏默认文件），将 `hf.download.shard_index_start` 与 `shard_index_end` 设为同一索引（例如 `0`），并把 `pipeline.delete_source_shard_tar` 设为 `false`，再执行 `pipeline_watchdog.py --config <副本> --once`。确认下载 / 渲染 / 编码日志正常后，恢复全量区间、删包策略与 `--once`。
