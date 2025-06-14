import json
import os

# --- Configuration ---

# 这是你之前T5设置中的TASK_PREFIX，可以作为LLaMA-Factory的"instruction"
# 或者你可以根据需要设计一个更通用的指令。
# 例如，如果你的prompt模板中已经包含了这部分，这里可以设为更通用的指令。
# 对于Alpaca格式，这个作为 instruction 很合适。
DEFAULT_INSTRUCTION = """你是一个专业的中文社交媒体内容分析助手。
你的任务是根据提供的输入文本，准确地抽取出其中所有相关的“仇恨言论四元组”。

每个四元组必须包含以下四个元素，并严格遵循指定的定义和格式输出：

1.  **评论对象 (Target):**
    *   定义：指输入文本中评论或提及的具体对象，可以是一个人或一个群体。
    *   特殊情况：如果输入文本没有明确的、具体的评论对象（例如，只是一句感叹或一般性陈述），则“评论对象”应设为 'NULL'。

2.  **论点 (Argument):**
    *   定义：指输入文本中针对“评论对象”所表达的关键观点、描述或攻击性言论的信息片段。

3.  **目标群体 (Targeted Group):**
    *   定义：指“评论对象”和“论点”所共同指向的、可能受到仇恨言论影响的群体。
    *   分类：
        *   当“是否仇恨”判断为 'hate' 时，“目标群体”必须是以下预定义类别之一：'地域'、'种族'、'性别'、'LGBTQ'、'其他'。
        *   当“是否仇恨”判断为 'non-hate' 时（包括非仇恨文本及不含特定上述群体的一般攻击性言论），“目标群体”也应设为 'non-hate'。

4.  **是否仇恨 (Hateful):**
    *   定义：判断由“评论对象”和“论点”组成的言论是否构成了对特定“目标群体”的仇恨。
    *   取值：应设为 'hate' 或 'non-hate'。

输出格式规范（请极度严格遵守，任何空格或顺序的偏差都可能导致错误）：
*   四元组顺序：严格按照 '评论对象 | 论点 | 目标群体 | 是否仇恨 [END]' 的顺序。
*   元素分隔符：每个元素之间必须使用 ' | ' (即：空格 竖线 空格) 进行分隔。
*   四元组结束符：每个四元组必须以 ' [END]' (即：空格 后紧跟 '[END]') 结尾。
*   多四元组分隔符：如果一条输入文本中可以抽取出多个四元组，那么不同的四元组之间必须使用 ' [SEP] ' (即：空格 后紧跟 '[SEP]' 再紧跟一个空格) 进行分隔。

请仔细分析输入文本，并生成符合上述所有要求的四元组字符串。"""
# 或者简单点，如果你在LLaMA-Factory的模板中处理更复杂的指令结构：
# DEFAULT_INSTRUCTION = "抽取仇恨言论四元组："


# 输入文件 (你的原始 train/dev JSON 文件)
# 确保这些路径是正确的
INPUT_TRAIN_FILE = "data/train_new.json"
# 假设你的验证集文件叫 dev.json，如果叫 val.json 请修改
INPUT_DEV_FILE = "data/dev.json"

# 输出文件 (JSONL 格式，用于 LLaMA-Factory)
# 建议将转换后的文件放在一个新的子目录中，例如 'data_jsonl'
OUTPUT_DIR_JSONL = "data_jsonl" # 输出目录名
OUTPUT_TRAIN_JSONL = os.path.join(OUTPUT_DIR_JSONL, "hate_speech_sft_train.jsonl")
OUTPUT_DEV_JSONL = os.path.join(OUTPUT_DIR_JSONL, "hate_speech_sft_dev.jsonl")

# --- Conversion Function ---

def convert_json_to_jsonl_for_sft(input_file_path, output_file_path, instruction_text):
    """
    将你的特定JSON格式数据转换为适用于LLaMA-Factory SFT的JSONL格式 (Alpaca风格)。
    输出JSONL中的每一行将是:
    {"instruction": "...", "input": "...", "output": "..."}
    """
    if not os.path.exists(input_file_path):
        print(f"错误: 输入文件未找到: {input_file_path}")
        return 0

    # 如果输出目录不存在，则创建它
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    converted_count = 0
    skipped_count = 0
    with open(input_file_path, 'r', encoding='utf-8') as infile, \
         open(output_file_path, 'w', encoding='utf-8') as outfile:

        original_data = json.load(infile) # 你的原始文件是整个列表是一个JSON对象

        for item in original_data:
            # 确保必要的字段存在
            if "content" not in item or "output" not in item:
                print(f"警告: 跳过记录 (ID: {item.get('id', '未知')}), 因为缺少 'content' 或 'output' 字段。")
                skipped_count += 1
                continue

            # 构建新的JSONL记录
            # LLaMA-Factory的SFT阶段通常使用类似Alpaca的格式
            # "instruction" + "input" 会被模板组合成最终的prompt
            # "output" 是模型需要学习生成的内容
            record = {
                "instruction": instruction_text,
                "input": item["content"],       # 原始的评论内容
                "output": item["output"]        # 原始的四元组标注字符串
            }
            outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
            converted_count += 1

    print(f"成功从 '{input_file_path}' 转换 {converted_count} 条记录到 '{output_file_path}'.")
    if skipped_count > 0:
        print(f"跳过了 {skipped_count} 条记录。")
    return converted_count

# --- Main Execution ---
if __name__ == "__main__":
    print("开始将数据转换为LLaMA-Factory SFT所需的JSONL格式...")

    # 转换训练数据
    print(f"\n处理训练数据: {INPUT_TRAIN_FILE}")
    convert_json_to_jsonl_for_sft(INPUT_TRAIN_FILE, OUTPUT_TRAIN_JSONL, DEFAULT_INSTRUCTION)

    # 转换验证数据 (如果存在)
    if os.path.exists(INPUT_DEV_FILE):
        print(f"\n处理验证数据: {INPUT_DEV_FILE}")
        convert_json_to_jsonl_for_sft(INPUT_DEV_FILE, OUTPUT_DEV_JSONL, DEFAULT_INSTRUCTION)
    else:
        print(f"\n警告: 验证文件 '{INPUT_DEV_FILE}' 未找到。跳过其转换。")

    print("\n数据转换完成。")
    print(f"输出的JSONL文件位于: '{OUTPUT_DIR_JSONL}/' 目录下")
    print("请确认这些文件 (例如 hate_speech_sft_train.jsonl, hate_speech_sft_dev.jsonl)")
    print("在LLaMA-Factory的 dataset_info.json 文件和训练脚本中被正确引用。")
    print(f"例如，在 dataset_info.json 中对应的 'file_name' 应设为 '{os.path.basename(OUTPUT_TRAIN_JSONL)}' 等。")