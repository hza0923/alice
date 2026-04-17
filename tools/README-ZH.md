# Tools 模块

`tools/` 目录已按功能分组：

- `tools/profiling/`：采集与 registry 构建
- `tools/scheduler/`：策略求解与解析基准
- `tools/runtime/`：proto 生成与任务提交
- `tools/simulation/`：DES 与本机多进程集群模拟
- `tools/transport/`：传输 A/B 基准（占位）

## 建议使用顺序

1. Capture  
   `python tools/profiling/capture_split_module_profiles.py --help`
2. Build registry  
   `python tools/profiling/build_registry.py --help`
3. Schedule strategy  
   `python tools/scheduler/solve_strategy.py --help`
4. Runtime / Simulator  
   `python tools/runtime/submit_task_to_master.py --help`  
   `python tools/simulation/run_pipeline_des_sim.py --help`  
   `python tools/simulation/run_cluster_sim.py --help`

## 兼容性

仓库仍保留扁平路径入口（如 `tools/build_registry.py`），以兼容已有脚本；新文档统一采用分组路径。
