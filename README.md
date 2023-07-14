### 介绍
Baichuan-13B 是由百川智能继 Baichuan-7B 之后开发的包含 130 亿参数的开源可商用的大规模语言模型，在权威的中文和英文 benchmark 上均取得同尺寸最好的效果。Baichuan-13B 有如下几个特点：
1. 更大尺寸、更多数据：Baichuan-13B 在 Baichuan-7B 的基础上进一步扩大参数量到 130 亿，并且在高质量的语料上训练了 1.4 万亿 tokens，超过 LLaMA-13B 40%，是当前开源 13B 尺寸下训练数据量最多的模型。支持中英双语，使用 ALiBi 位置编码，上下文窗口长度为 4096。
2. 同时开源预训练和对齐模型：预训练模型是适用开发者的『 基座 』，而广大普通用户对有对话功能的对齐模型具有更强的需求。因此本次开源同时发布了对齐模型（Baichuan-13B-Chat），具有很强的对话能力，开箱即用，几行代码即可简单的部署。
3. 更高效的推理：为了支持更广大用户的使用，本次同时开源了 int8 和 int4 的量化版本，相对非量化版本在几乎没有效果损失的情况下大大降低了部署的机器资源门槛，可以部署在如 Nvidia 3090 这样的消费级显卡上。

[Baichuan-13B官方链接](https://github.com/baichuan-inc/Baichuan-13B)

### 环境准备
#### 1. 下载微调项目
```sh
git clone https://github.com/mark1879/Baichuan-13B-Finetuning.git
```

#### 2. 下载 baichuan-13B 大模型
```sh
# 在根目录下新建 baichuan-inc 保存大模型
cd Baichuan-13B-Finetuning
mkdir baichuan-inc
cd baichuan-inc

git lfs install

# 官方微调过（指令对齐）
git clone https://huggingface.co/baichuan-inc/Baichuan-13B-Chat

# 预训练大模型（未经过微调）
#git clone https://huggingface.co/baichuan-inc/Baichuan-13B-Base
```

#### 3. 安装 python 库
```sh
pip3 install -r requirements.txt
```

#### 4. 硬件要求
- LoRA 显存：>= 32G
- QLoRA 显存：>= 12G

### 微调
#### 1. 准备训练数据
`data` 目录下存储了训练数据，可根据业务需要自行准备数据，数据格式如下：
- **instruction**：任务指令，不能为空。
- **input**：任务输入，可为空。如果不为空，项目内部处理训练数据时，会将 instruction、input 拼接在一起作为任务的输入。
- **output**：任务输出，不能为空。
```json
[
  {
    "instruction": "什么是三原色？",
    "input": "",
    "output": [
      "三原色是红、蓝、黄。这些颜色被称为原色，因为它们不能通过混合其他颜色来创建，所有其他颜色都可以通过将它们按不同比例组合而成。在用于光的加色系统中，原色是红色、绿色和蓝色 (RGB)。",
      "红色、黄色和绿色。"
    ]
  },
  {
    "instruction": "写一段关于一个人特点的描述",
    "input": "姓名：阿比盖尔\n喜欢的东西：动作电影、法国菜、热衷于商业",
    "output": "阿比盖尔是一个冒险的灵魂，喜欢看动作电影和吃法国美食。她对商业充满热情，并努力培养。她阅读投资新闻，密切关注股市。每当有机会出现，阿比盖尔总是迅速行动，不会犹豫利用她的商业知识。她是那种喜欢经历商业起伏、善于追求交易并与志同道合的人交流的人。"
  }
]
```

#### 2. Lora 微调

- **CUDA_VISIBLE_DEVICES=0**: &nbsp;&nbsp;单卡运行。
- **do_train**: &nbsp;&nbsp;是否执行训练。
- **model_name_or_path**: &nbsp;&nbsp;预训练模型路径。
- **dataset_dir**: &nbsp;&nbsp;训练数据存储目录。
- **dataset**: &nbsp;&nbsp;训练数据集名称，可在 data/dataset_info.json 中增加自定义数据集。
- **output_dir**: &nbsp;&nbsp;微调后的模型保存路径。
- **source_prefix**:&nbsp;&nbsp;训练时每个输入序列添加的前缀，可为空。
- **max_source_length**: &nbsp;&nbsp;输入序列的最大长度，即 source_prefix + instruction + input 的长度。
- **max_target_length**: &nbsp;&nbsp;输出序列的最大长度，即 output 的长度。
- **per_device_train_batch_size**: &nbsp;&nbsp;用于训练的批处理大小。可根据 GPU 显存大小自行设置。
- **gradient_accumulation_steps**: &nbsp;&nbsp;梯度累加次数。
- **logging_steps**: &nbsp;&nbsp;多少步输出一次 log。
- **save_steps**: &nbsp;&nbsp;多少步保存一次参数。
- **learning_rate**: &nbsp;&nbsp;AdamW 优化器的初始学习率。
- **num_train_epochs**: &nbsp;&nbsp;训练轮数（若非整数，则最后一轮只训练部分数据）
- **plot_loss**: &nbsp;&nbsp;微调后绘制损失函数曲线，图片保存在 output_dir 中 。
- **fp16**: &nbsp;&nbsp;使用半精度（混合精度）训练。
- **lora_target**: &nbsp;&nbsp;大模型内将要进行 LoRA 微调的模块名称。
- **lora_rank**: &nbsp;&nbsp;LoRA 微调中的秩大小。
- **padding_side**: &nbsp;&nbsp; pad对齐方式，左对齐或者右对齐。
```sh
CUDA_VISIBLE_DEVICES=0 python finetune_lora.py \
    --do_train \
    --model_name_or_path baichuan-inc/Baichuan-13B-Chat \
    --dataset_dir data \
    --dataset alpaca_gpt4_zh \
    --output_dir baichuan_lora_checkpoint \
    --source_prefix ""  \
    --max_source_length 256 \
    --max_target_length 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 1.0 \
    --plot_loss \
    --fp16  \
    --lora_target W_pack \
    --lora_rank 8  \
    --padding_side right
```

#### 3. 测试微调后的模型
- **CUDA_VISIBLE_DEVICES=0**: &nbsp;&nbsp;单卡运行。
- **do_eval**: &nbsp;&nbsp;是否执行测试。
- **model_name_or_path**: &nbsp;&nbsp;预训练模型路径。
- **checkpoint_dir**: &nbsp;&nbsp;微调模型路径。
- **dataset_dir**: &nbsp;&nbsp;测试数据存储目录。
- **dataset**: &nbsp;&nbsp;测试数据集名称，可在 data/dataset_info.json 中增加自定义数据集。
- **output_dir**: &nbsp;&nbsp;测试结果保存路径。
- **per_device_eval_batch_size**：&nbsp;&nbsp;测试数据的批处理大小。可根据 GPU 显存大小，自行设置。
- **predict_with_generate**: &nbsp;&nbsp;是否生成序列用于计算 ROUGE 或 BLEU 分数。
- **padding_side**: &nbsp;&nbsp; pad对齐方式，左对齐或者右对齐。

```sh
CUDA_VISIBLE_DEVICES=0 python finetune_lora.py \
    --do_eval \
    --model_name_or_path baichuan-inc/Baichuan-13B-Chat \
    --checkpoint_dir baichuan_lora_checkpoint \
    --dataset_dir data \
    --dataset alpaca_gpt4_zh_test \
    --output_dir baichuan_lora_eval_result \
    --per_device_eval_batch_size 1 \
    --predict_with_generate  \
    --padding_side right
```

#### 4. 与大模型对话

- **model_name_or_path**: &nbsp;&nbsp;预训练模型路径。
- **checkpoint_dir**: &nbsp;&nbsp;微调模型路径。
```sh
python cli_demo.py \
    --model_name_or_path baichuan-inc/Baichuan-13B-Chat \
    --checkpoint_dir baichuan_lora_checkpoint
```

### 扩展
#### 1. QLora 微调
```sh
# /usr/local/cuda-xx.x/ 是本机 cuda 安装路径
export LD_LIBRARY_PATH=/usr/local/cuda-12.0/lib64

pip3 install bitsandbytes
pip3 install scipy
pip3 install git+https://github.com/huggingface/peft.git
```

`sh train.sh` 开启训练时增加 `--quantization_bit 4` 参数。 

#### 2. 更多参数设置
请参考 `config.py` 文件。

### Acknowledgement
本项目是基于 [LLaMA-Efficient-Tuning](https://github.com/hiyouga/LLaMA-Efficient-Tuning) 项目中的 LoRA 微调部分修改而来，在此表示感谢！
