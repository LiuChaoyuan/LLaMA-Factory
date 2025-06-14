# 中文仇恨言论四元组抽取研究 (课程作业)

本项目旨在针对中文社交媒体平台中的仇恨言论进行片段级四元组（评论对象、论点、目标群体、是否仇恨）抽取。主要探索并对比了两种基于预训练语言模型的方法：
1.  **基于 T5-base 模型的序列到序列微调**。
2.  **基于 Qwen1.5-1.8B-Chat 模型的 LoRA 参数微调** (利用 LLaMA-Factory 框架)。

> 基于 Qwen1.5-1.8B-Chat 模型的 LoRA 参数微调中，未合并的模型文件在saves文件夹下。
> 基于 T5-base 模型的序列到序列微调的模型文件在[百度网盘链接](https://pan.baidu.com/s/1PeXVNAOFAA5Whw-FDwlcQg?pwd=u9ym)提取码: u9ym

## 项目结构

本仓库基于 [LLaMA-Factory (原始仓库链接)](https://github.com/hiyouga/LLaMA-Factory)进行修改和扩展。

- `/` (根目录): 主要包含 LLaMA-Factory 的原始文件和结构。
- `/data/`: 存放 LLaMA-Factory 使用的数据集。
    - `dataset_info.json`: LLaMA-Factory 的数据集配置文件。
    - `hate_speech_sft_train.jsonl`: 转换后的训练数据 (JSONL格式)。
    - `hate_speech_sft_dev.jsonl`: 转换后的验证数据 (JSONL格式)。
- `/saves/`: LLaMA-Factory 训练模型的输出目录 (例如 LoRA 权重、检查点)。
- `/models/`: (建议) 存放合并后或最终用于推理的模型，例如 `Qwen1.5-1.8B-Chat-hate-speech-merged`。
- `/results/`: 存放模型预测结果 (例如 `.txt` 文件) 和评估指标记录。
- `/src/`: LLaMA-Factory 的核心代码。
    - `train.py`: LLaMA-Factory 的主训练脚本 (本项目主要使用此脚本)。
    - `export_model.py`: LLaMA-Factory 用于合并 LoRA 权重的脚本。
- `/t5_base/`: 包含使用 T5-base 模型进行四元组抽取的独立实现代码。
    - `main_t5.py`: T5 模型训练和预测的主入口脚本。
    - `train_t5.py`: T5 模型的训练逻辑。
    - `predict_t5.py`: T5 模型的预测逻辑。
    - `dataset_t5.py`: T5 模型的数据集处理类。
    - `utils_t5.py`: T5 模型的工具函数 (如评估指标计算)。
    - `config_t5.py`: T5 模型的主要配置文件。
    - `data/`
		- `train.json`: 原始训练数据。
	    - `train_new.json`: T5 使用的训练数据。        
        - `dev.json`: T5 使用的验证数据。
        - `test1.json`, `test2.json`: 原始测试数据。
    - `results/`: T5 模型的输出和评估结果。
- `convert_data_to_jsonl.py`: 用于将原始 JSON 数据转换为 LLaMA-Factory 所需 JSONL 格式的脚本。
- `README.md`: 本说明文件。
- `LICENSE`: LLaMA-Factory 的原始许可证文件
- `t5_base/LICENSE`: (可选) T5-base 部分代码的许可证文件。

## 环境搭建与依赖

### 1. 创建虚拟环境 (推荐)
```bash
conda create -n hate_speech_env python=3.10 # 推荐使用与LLaMA-Factory兼容的Python版本
conda activate hate_speech_env
```

### 2. 安装 LLaMA-Factory 依赖
请参照 LLaMA-Factory 官方文档进行安装。通常包括：
```bash
cd LLaMA-Factory # 进入 LLaMA-Factory 根目录
pip install -e .[qwen,torch,deepspeed,metrics] # 根据需要安装可选依赖，qwen是必须的
# 或者根据其 requirements.txt 文件安装
# pip install -r requirements.txt

# 可能需要单独安装 bitsandbytes 和 flash-attn (如果使用QLoRA和Flash Attention)
# pip install bitsandbytes accelerate
# pip install flash-attn --no-build-isolation # (需要合适的CUDA和PyTorch版本)
```
**注意**: 请确保你的 CUDA、PyTorch 版本与 `bitsandbytes` 和 `flash-attn` 的要求相兼容。

### 3. 安装 T5-base 部分依赖
```bash
cd ../t5_base # 进入 t5_base 目录 (假设从 LLaMA-Factory 根目录返回上一级再进入)
pip install torch transformers tqdm dataset  numpy 
```

## 数据准备

### 1. 原始数据
原始数据为 JSON 格式，包含 `id`, `content` (输入文本), `output` (目标四元组字符串) 字段。
- 训练数据: `t5_base/data/train_new.json`
- 验证数据: `t5_base/data/dev.json`
- 测试数据: `t5_base/data/test1.json`, `t5_base/data/test2.json` (无 `output` 字段)

### 2. 为 LLaMA-Factory (Qwen) 准备数据

LLaMA-Factory 的 SFT (Supervised Fine-Tuning) 任务通常使用 JSONL 格式，每行一个 JSON 对象，包含 `instruction`, `input`, `output` 字段。

**a. 运行数据转换脚本：**
我们提供了一个脚本 `convert_data_to_jsonl.py` (位于仓库根目录) 来将原始的 `train.json` 和 `dev.json` 转换为所需的 JSONL 格式。
```bash
python convert_data_to_jsonl.py
```
修改该脚本中的 `INPUT_TRAIN_FILE`, `INPUT_DEV_FILE` 和 `OUTPUT_DIR_JSONL` 变量以匹配你的实际路径。
该脚本会生成：
- `data_jsonl/hate_speech_sft_train.jsonl`
- `data_jsonl/hate_speech_sft_dev.jsonl`

**b. 移动并配置 LLaMA-Factory 数据：**
1.  将生成的 `hate_speech_sft_train.jsonl` 和 `hate_speech_sft_dev.jsonl` 文件移动到 LLaMA-Factory 的数据目录：`LLaMA-Factory/data/`。
2.  修改或创建 `LLaMA-Factory/data/dataset_info.json` 文件，添加如下条目 (确保键名与你后续训练命令中的 `--dataset` 和 `--eval_dataset` 参数值一致)：

    ```json
    {
      "hate_speech_sft": { // 训练数据集的键名
        "file_name": "hate_speech_sft_train.jsonl",
        "formatting": "alpaca", // 明确指定格式为 alpaca
        "columns": {
          "prompt": "instruction", // jsonl文件中的 "instruction" 字段
          "query": "input",        // jsonl文件中的 "input" 字段
          "response": "output"     // jsonl文件中的 "output" 字段
        }
      },
      "hate_speech_sft_dev": { // 验证数据集的键名
        "file_name": "hate_speech_sft_dev.jsonl",
        "formatting": "alpaca",
        "columns": {
          "prompt": "instruction",
          "query": "input",
          "response": "output"
        }
      }
      // ... 其他数据集配置 ...
    }
    ```

## 运行实验

### 方法一：基于 T5-base 的序列到序列微调

1.  **修改配置文件**:
    进入 `t5_base/` 目录，根据需要修改 `config_t5.py` 中的参数，例如：
    - `MODEL_NAME`: 使用的 T5 预训练模型 (例如 `mengzi-t5-base`)。
    - `TRAIN_FILE`, `DEV_FILE`, `TEST1_FILE`, `TEST2_FILE`: 数据文件路径。
    - `OUTPUT_DIR`: 模型和结果的输出路径。
    - `NUM_TRAIN_EPOCHS`, `PER_DEVICE_TRAIN_BATCH_SIZE`, `LEARNING_RATE` 等训练超参数。

2.  **启动训练**:
    在 `t5_base/` 目录下运行：
    ```bash
    python main_t5.py --do_train
    ```
    训练过程中，模型检查点和日志会保存在 `config_t5.py` 中定义的 `OUTPUT_DIR` 下。评估指标 (hard/soft F1) 会在每个 epoch 结束后打印并记录到 `eval_results.txt`。

3.  **进行预测**:
    训练完成后，最佳模型会保存在 `OUTPUT_DIR` 下的 `best_model_after_eval/` (如果 `LOAD_BEST_MODEL_AT_END=True`)。
    在 `t5_base/` 目录下运行 (以 test1.json 为例)：
    ```bash
    python main_t5.py --do_predict --test_file test1
    ```
    预测结果将保存为 `OUTPUT_DIR/predictions_test1.json` (JSON格式，如需TXT请修改 `predict_t5.py` 中的保存逻辑)。

### 方法二：基于 Qwen1.5-1.8B-Chat 的 LoRA 微调 (使用 LLaMA-Factory)

1.  **准备本地模型 (可选，如果服务器无法访问 Hugging Face Hub)**:
    如果服务器网络不佳，请先将 `Qwen/Qwen1.5-1.8B-Chat` 模型完整下载到服务器本地路径，例如 `/root/autodl-tmp/hf_models/Qwen1.5-1.8B-Chat`。

2.  **修改训练参数/配置文件**:
    LLaMA-Factory 支持通过命令行参数或 YAML 配置文件进行训练。
    **a. 使用命令行参数 (示例)：**
    在 `LLaMA-Factory/` 根目录下运行。请根据你的实际情况调整参数，特别是 `--model_name_or_path` (如果使用本地模型，请填写本地路径)。

    ```bash
    # 确保你已完成数据准备步骤中的 JSONL 转换和 dataset_info.json 配置
    cd /path/to/LLaMA-Factory # 进入 LLaMA-Factory 根目录

    CUDA_VISIBLE_DEVICES=0 python src/train.py \
        --model_name_or_path /root/autodl-tmp/hf_models/Qwen1.5-1.8B-Chat \ # 或 HuggingFace Hub 上的模型名
        --stage sft \
        --do_train \
        --do_eval \
        --dataset hate_speech_sft \ # 必须与 dataset_info.json 中的键名一致
        --eval_dataset hate_speech_sft_dev \ # 必须与 dataset_info.json 中的键名一致
        --finetuning_type lora \
        --lora_target all \ # 或 'q_proj,v_proj,o_proj,k_proj'
        --lora_rank 32 \
        --lora_alpha 64 \
        --quantization_bit 4 \
        --quantization_type nf4 \
        --double_quantization true \
        --upcast_layernorm true \
        --output_dir saves/Qwen1.5-1.8B-Chat/qlora/sft_hate_speech \
        --per_device_train_batch_size 8 \ # 根据你的4090显存尝试增加 (例如 16, 24)
        --gradient_accumulation_steps 4 \ # 对应减小
        --learning_rate 5.0e-5 \ # LoRA 通常用 1e-4 到 5e-5
        --num_train_epochs 5.0 \ # 建议从较少 epoch 开始
        --lr_scheduler_type cosine \
        --warmup_ratio 0.1 \
        --logging_strategy epoch \
        --eval_strategy epoch \ # 或 steps
        # --eval_steps 100 \ # 如果 eval_strategy 是 steps
        --save_strategy epoch \ # 或 steps
        # --save_steps 100 \ # 如果 save_strategy 是 steps
        --bf16 true \ # RTX 4090 支持 bf16
        --plot_loss true \
        --load_best_model_at_end true \
        --metric_for_best_model eval_loss \ # LLaMA-Factory 默认会计算 eval_loss
        --greater_is_better false \
        --template qwen \
        --cutoff_len 1280 \ # 或根据你的数据调整，例如 768
        --max_samples 4000 \ # 如果你只想用一部分数据训练
        --overwrite_cache true \ # 第一次运行时建议为true，之后可以为false
        --preprocessing_num_workers 16 \
        --dataloader_num_workers 8 # 根据CPU核心数调整
        # --flash_attn auto # 如果已安装 flash-attn，可以尝试开启
    ```
    **b. 使用 YAML 配置文件 (推荐用于复杂配置管理)：**
    1.  在 LLaMA-Factory 的 `examples/train_lora/` 目录下，可以找到llama3_lora_sft.yaml,这是一个YAML 配置文件。
    2.  修改这个配置文件。
    3.  运行训练：
        ```bash
        llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml
        ```

3.  **合并 LoRA 权重 (可选但推荐用于推理)**:
    训练完成后，LoRA 权重会保存在 `--output_dir` 下的 checkpoint 目录中。使用 `export_model.py` 脚本将 LoRA 权重合并到基础模型中。
	在 LLaMA-Factory 的 `examples/merge_lora/` 目录下，可以找到llama3_lora_sft.yaml,这是一个YAML 配置文件。
	修改上述配置文件，特别注意**原始模型的路径**和LoRA权重的路径。
	修改完成后按照下面的脚本合并模型。
    ```bash
    llamafactory-cli export examples/merge_lora/llama3_lora_sft.yaml
    ```

4.  **使用合并后的模型进行推理**:
    使用 `predict_qwen.py` 脚本 (你需要根据前面对话中我们讨论的内容来编写或调整此脚本)。
    确保 `predict_qwen.py` 中的 `MODEL_PATH` 指向你合并后的模型路径 (例如 `/root/models/Qwen1.5-1.8B-Chat-hate-speech-merged`)。
    ```bash
    python predict_qwen.py # 假设 predict_qwen.py 内部已配置好测试文件和输出文件路径
    ```
    该脚本会将预测结果保存为 TXT 文件，每行一个预测的四元组字符串。

## 结果与分析
- T5-base 模型在验证集上最终的 F1-avg 为 0.2628。
- Qwen1.5-1.8B-Chat (LoRA) 模型在验证集上最终的 F1-avg 为 0.2542。
- 详细的评估指标记录分别在 `t5_base/results/eval_results.txt` 和 LLaMA-Factory 的输出目录 
- 预测的输出文本文件保存在 `results/` 目录下。

## 主要改动和贡献点
- 独立实现了基于 T5-base 的仇恨言论四元组抽取模型，包括数据处理、训练、预测和评估全流程。
- 利用 LLaMA-Factory 框架，成功对 Qwen1.5-1.8B-Chat 模型进行了 LoRA 参数高效微调，并应用于本任务。
- 对数据进行了预处理，将其转换为适用于不同框架的格式。
- 设计并迭代了用于指导模型进行信息抽取的 Prompt (Instruction)。
- 对比分析了两种不同规模和微调策略的模型在本任务上的性能表现。

## 致谢
- 本项目基于 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 开源项目进行。感谢其作者的贡献。
- 模型训练和实验基于 Hugging Face Transformers 库以及 Qwen、T5 等预训练模型。

## 版权与许可证
- LLaMA-Factory 部分遵循其原始的[Apache 2.0 License] (LICENSE)。
- `t5_base/` 目录下的代码遵循Apache 2.0 License。
