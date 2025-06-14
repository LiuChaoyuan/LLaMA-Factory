import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from tqdm import tqdm
import os # 用于路径操作

# --- 配置 ---
# 确保这个路径是服务器上模型的准确绝对路径
MODEL_PATH = "/root/models/lr5-4epoch20" 

# 输入的测试文件名 (假设与脚本在同一目录或指定完整路径)
# 例如，如果 test1.json 在 ./data/ 目录下，就用 "data/test1.json"
TEST_FILE_BASENAME = "test1.json" # 或者 "test2.json"
TEST_FILE_PATH = os.path.join("data", TEST_FILE_BASENAME) # 假设在 data 目录下

# 输出文件名 (TXT格式)
OUTPUT_DIR = "results"
OUTPUT_TXT_FILE = os.path.join(OUTPUT_DIR, f"lr5-5epoch20_{os.path.splitext(TEST_FILE_BASENAME)[0]}.txt")

# 与微调时数据转换脚本中的 DEFAULT_INSTRUCTION 保持一致
# 这是你之前提供的详细指令
SFT_INSTRUCTION =  """你是一个专业的中文社交媒体内容分析助手。
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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 200  # 与微调时 max_target_length 或期望输出长度一致
DEBUG_SAMPLES = None   # 设置为 None 处理所有样本，或设置为数字进行调试，例如 10

# --- 主函数 ---
def predict():
    print(f"使用的设备: {DEVICE}")
    
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 加载模型和分词器 ---
    print(f"正在从以下路径加载分词器: {MODEL_PATH}")
    # 对于Qwen1.5，通常需要 trust_remote_code=True
    # use_fast=False 可能对某些复杂tokenization或特殊token更稳定，但True通常更快
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, use_fast=True)
    
    # Qwen tokenizer 可能没有显式的 pad_token。如果需要，可以设置为 eos_token。
    if tokenizer.pad_token_id is None:
        print("分词器的 pad_token 未设置，将其设置为 eos_token。")
        tokenizer.pad_token_id = tokenizer.eos_token_id
        # 对于模型配置也可能需要更新（如果模型内部使用了 pad_token_id）
        # model.config.pad_token_id = tokenizer.pad_token_id 

    print(f"正在从以下路径加载模型: {MODEL_PATH}")
    # 对于QLoRA微调后合并的模型，通常不需要再指定 quantization_config
    # device_map="auto" 会自动将模型分配到可用设备
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    print("模型和分词器加载完成。")

    # --- 加载测试数据 ---
    print(f"正在加载测试数据从: {TEST_FILE_PATH}")
    try:
        with open(TEST_FILE_PATH, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print(f"错误: 测试文件未找到 {TEST_FILE_PATH}")
        return
    except json.JSONDecodeError:
        print(f"错误: 测试文件 {TEST_FILE_PATH} 不是有效的JSON格式。")
        return

    if DEBUG_SAMPLES:
        raw_data = raw_data[:DEBUG_SAMPLES]
        print(f"调试模式：仅处理前 {DEBUG_SAMPLES} 条样本。")
    
    test_inputs = []
    for item in raw_data:
        test_inputs.append({"id": item["id"], "content": item["content"]})
    
    # --- 预测 ---
    predicted_outputs = []
    print(f"开始对 {len(test_inputs)} 条样本进行预测...")
    
    # 使用与LLaMA-Factory微调时相同的模板（或等效的prompt构造方式）
    # 对于Qwen-Chat模型和Alpaca格式SFT数据，模板通常如下：
    # System Prompt (可选)
    # User: instruction + input
    # Assistant: (模型需要生成的部分)
    
    # Qwen1.5 Chat 的模板通常是 <|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n
    # 这里的 SFT_INSTRUCTION 是用户指令，item["content"] 是用户输入
    
    for item_data in tqdm(test_inputs, desc="预测中"):
        # 构造符合Qwen Chat模板的输入
        # 注意：这里的 prompt 构造方式需要与 LLaMA-Factory 微调时 `--template qwen` (或你实际用的template)
        # 所生成的最终输入格式尽可能一致。
        # 如果 LLaMA-Factory 的 qwen 模板将你的SFT_INSTRUCTION视为system prompt的一部分，
        # 或者将SFT_INSTRUCTION和item["content"]都放在user部分，你需要模拟这种结构。
        
        # 假设 qwen 模板将 instruction 和 input 都放在 user 部分，并且有一个通用的 system prompt
        # (你需要确认 LLaMA-Factory 中 'qwen' template 的具体行为)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."}, # 通用系统提示
            {"role": "user", "content": SFT_INSTRUCTION + "\n输入文本：\n" + item_data["content"]} # 组合指令和实际输入
        ]
        
        try:
            # 使用 tokenizer.apply_chat_template 来正确格式化输入
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True # 重要：为生成添加 <|im_start|>assistant\n
            )
        except Exception as e:
            print(f"错误：应用聊天模板失败，可能是tokenizer缺少chat_template配置: {e}")
            print(f"当前tokenizer的chat_template: {tokenizer.chat_template}")
            # 如果tokenizer没有预设的chat_template，你可能需要手动设置或使用更简单的prompt格式
            # 备用简单prompt (如果apply_chat_template失败或不适用):
            # prompt_text = SFT_INSTRUCTION + "\n输入文本：\n" + item_data["content"] + "\n输出："
            # print("回退到简单prompt格式。")
            # 这里我们先假设 apply_chat_template 会成功
            predicted_outputs.append("") # 添加空串以保持输出行数一致
            continue


        model_inputs = tokenizer([prompt_text], return_tensors="pt", padding=False).to(model.device) # padding=False 因为是单条预测

        with torch.no_grad():
            generated_ids = model.generate(
                model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id, # 明确指定eos_token_id
                # 可以考虑添加其他生成参数，如 num_beams, do_sample, top_p, temperature
                # 例如: num_beams=3, do_sample=False (如果想要更确定的输出)
            )
        
        # 解码时跳过prompt部分
        # generated_ids[0] 是整个序列，model_inputs.input_ids.shape[1] 是prompt的长度
        response_ids = generated_ids[0][model_inputs.input_ids.shape[1]:]
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        
        predicted_outputs.append(response_text)

    # --- 保存结果为 TXT 文件 ---
    print(f"\n正在将预测结果保存到: {OUTPUT_TXT_FILE}")
    try:
        with open(OUTPUT_TXT_FILE, 'w', encoding='utf-8') as f:
            for line in predicted_outputs:
                f.write(line + "\n")
        print("预测结果成功保存。")
    except Exception as e:
        print(f"错误: 保存预测结果失败: {e}")

if __name__ == "__main__":
    predict()