import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel
from PIL import Image
import requests

# 1. 你的 LoRA 权重路径 (请根据你的实际文件夹修改这里！)
# 通常在 runs/ 下面，找到那个以 checkpoint-5000 结尾或者最新的文件夹
# 示例路径，你需要改成你机器上真实存在的那个路径：
ADAPTER_PATH = "/root/gpufree-data/project/datadisk/adapter_tmp/openvla-7b+libero_spatial_no_noops+b16+lr-0.0005+lora-r16+dropout-0.0--image_aug" 
# 注意：要把上面这个路径改成你 ls runs/ 看到的那个真实路径

# 2. 基础模型路径
BASE_MODEL_PATH = "openvla/openvla-7b" 

print("正在加载基础模型...")
# 加载处理器
processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
# 加载基础模型 (以 bfloat16 加载节省显存)
model = AutoModelForVision2Seq.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.bfloat16, 
    low_cpu_mem_usage=True, 
    trust_remote_code=True
).to("cuda")

print(f"正在挂载微调权重: {ADAPTER_PATH}")
# 加载你训练好的 LoRA "外挂"
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.merge_and_unload() # 这一步在显存里合并，方便推理

print("模型加载成功！准备测试...")

# 3. 搞一张测试图片 (这里用一张网上的示例图，或者你可以换成你自己的本地图片)
print("⚠️由于网络限制，正在使用本地生成的测试图片...")
# 创建一张 224x224 的纯黑图片，足以用来测试模型通道是否打通
image = Image.new('RGB', (224, 224), color='black')

# 4. 输入一条指令 (Prompt)
prompt = "In: What action should the robot take to [pick up the yellow block]?\nOut:"

# 5. 处理输入
inputs = processor(prompt, image).to("cuda", dtype=torch.bfloat16)

# 6. 预测动作
print("正在预测动作...")
action = model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

print("\n====== 预测结果 ======")
print(f"指令: {prompt}")
print(f"模型生成的动作向量: {action}")
print("======================")

