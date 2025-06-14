## 风灵月影安全大模型
### 前言
ChatGPT开启了大语言模型发展的新方向，各大互联网巨头纷纷进入赛道。各大高校也加大对LLM的研发应用。在通用大语言模型领域OpenAI的统治地位暂时无可撼动；因此针对特定领域（Specific-Domain）的大语言模型是发展的必然趋势。目前医疗、教育、金融，法律领域已逐渐有了各自的模型，但网络安全领域迟迟没有模型发布。

无独有偶的是，微软也有类似定位产品——Microsoft Security Copilot，探索将自然语言处理的大语言模型引入到网络安全乃至安全审计领域这一条路线；

风灵月影安全大模型目标是为安全审计和网络防御提供强大的自然语言处理能力。它具备分析恶意代码、检测网络攻击、预测安全漏洞等功能，为安全专业人员提供有力的支持。
### 训练准备
#### 训练平台
基于colab平台进行训练[https://colab.research.google.com/drive/1D4EcAHGGQrnND5gDssz8aSZ_s8AOksbu?usp=sharing](url)

#### 训练集
数据集主要来自于Github、Kaggle、安全网站、公开的安全漏洞数据集组成，随后经过清洗、数据增强等来构造对话数据。数据集严格按照Alpaca模型数据集格式组织，分为Instruction，input，output三部分，我们规定在output输出中应当包含对具体内容的分析（analysis），安全评级（label），安全风险（risk），以及对应的解决方案（solution）。
SQL Inject：
![](https://cmd.dayi.ink/uploads/upload_c66c36f2db9769efca0f355a44b04201.png)

XSS：
![](https://cmd.dayi.ink/uploads/upload_7adfaf09d5fc36f368fa5c832d0bc760.png)

Bash：
![](https://cmd.dayi.ink/uploads/upload_8021d7e17a22453dec0724e9a73d6268.png)

Python：
![](https://cmd.dayi.ink/uploads/upload_f3216b88bc2c3920d68c70174bec459b.png)



#### 训练模型
使用Meta Llama 3模型，Llama3是在两个定制的24K的GPU集群上、基于超过 15T token 的数据上进行了训练，其中代码数据相当于Llama2的4倍。从而产生了迄今为止最强大的Llama模型。Llama3支持8K上下文长度是Llama2的两倍。
![](https://cmd.dayi.ink/uploads/upload_419850ad04955bcb3a8d407aaf1ec19e.png)

由清华大学基础模型研究中心联合中关村实验室研制的SuperBench大模型综合能力评测平台，基于语义、对齐、代码、安全和智能体5项大模型原生评测基准，展开开放性、动态性、科学性和权威性的大模型综合能力评测，率先剖析Llama 3模型能力。
![](https://cmd.dayi.ink/uploads/upload_86733b2bc7c9753f9424eb7b2285d1c9.png)

在代码编写能力、人类对齐能力、安全和价值观三项评测中，Llama 3-70B均排在第七名，超过大部分国内大模型，只落败于GLM-4和文心一言4.0，Llama 3-8B排名相对靠后，考虑到模型参数量的差异，Llama 3-70B整体表现较好。
![](https://doc.renil.cc/uploads/4e5daea6-28a2-4595-84c8-f43e1029a4f9.png)

Llama 3 提供两个版本：8B 版本适合在消费级 GPU 上高效部署和开发；70B 版本则专为大规模 AI 应用设计。每个版本都包括基础和指令调优两种形式。
本大模型基于llama-3-8b-Instruct版本，使用Unsloth优化版本unsloth/llama-3-8b-Instruct-bnb-4bit进行微调
Unsloth 是一个开源的大模型训练加速项目，使用 OpenAI 的 Triton 对模型的计算过程进行重写，大幅提升模型的训练速度，降低训练中的显存占用。Unsloth 能够保证重写后的模型计算的一致性，实现中不存在近似计算，模型训练的精度损失为零。Unsloth 支持绝大多数主流的 GPU 设备，包括 V100, T4, Titan V, RTX 20, 30, 40x, A100, H100, L40 等，支持对 LoRA 和 QLoRA 的训练加速和高效显存管理，支持 Flash Attention。
![](https://cmd.dayi.ink/uploads/upload_a050535c204608c36ad9e4662a25e267.png)

使用Lora进行微调，LoRA 是一种微调方法，它利用量化和低秩适配器来有效地减少计算需求和内存占用。
虽然 LLM，或者说在 GPU 上被训练的模型的随机性不可避免，但是采用 LoRA 进行多次实验，LLM 最终的基准结果在不同测试集中都表现出了惊人的一致性。对于进行其他比较研究，这是一个很好的基础。


配置如下
```
from trl import SFTTrainer
from transformers import TrainingArguments

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, #  建议 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth", # 检查点，长上下文度
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # 可以让短序列的训练速度提高5倍。
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,  # 微调步数
        learning_rate = 2e-4, # 学习率
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)
```
#### 开始训练
![](https://cmd.dayi.ink/uploads/upload_58021a1d5e088e48c2b6a8e053be3f73.png)
可以看到损失函数在逐步下降
#### 微调模型测试
![](https://cmd.dayi.ink/uploads/upload_9519681868f1e925d878064abc434a7c.png)

#### 保存模型，合并量化成4位gguf保存，大小约为 5g
![](https://doc.renil.cc/uploads/51dcabef-06f8-42fc-99c6-9a50c40dea04.png)
#### 模型可以通过GPT4all进行使用
![](https://cmd.dayi.ink/uploads/upload_8ee08b35670bb3938fcc1047d8aadeb9.png)

### TODO
目前大模型仍存在中文支持不足、回答错误等情况，后续将会调整数据集、调整LORA微调参数进行改善，未来会将大模型集成在其余设备中以方便使用

## 阶段性成果
### 基于unsloth微调mistral-7b-v0.3
![](https://cmd.dayi.ink/uploads/upload_d566029fafc594d9f24cb38d1f9628a0.png)
![](https://cmd.dayi.ink/uploads/upload_7dcf47ffe152e186449509589bed930c.png)


### 基于unsloth微调Llama3-Chinese-8B-Instruct模型
![](https://cmd.dayi.ink/uploads/upload_373c02d0fb61e93207f09548f9da3bad.png)
![](https://cmd.dayi.ink/uploads/upload_64c8d2bf614de733090ac54f4132d256.png)
![](https://cmd.dayi.ink/uploads/upload_ab2b768a4b9d5c30cb6adacdac21c8f7.png)



### 使用GPT4all或者ollama运行大模型
输入```ollama list``` 查看导入的模型
![](https://cmd.dayi.ink/uploads/upload_e9574915e5573fed174eeeb3ecc0ace8.png)

输入 ```ollama run llama3-cn```运行模型
![](https://cmd.dayi.ink/uploads/upload_34f32ec0ba390f20c63dd7373256e5df.png)
在colab中调用
![](https://cmd.dayi.ink/uploads/upload_915f1328174f714264fdfd31ca5f8f15.png)
### 微调前后对比
![](https://cmd.dayi.ink/uploads/upload_120ff5e256e2f8599bb61559db4d7a24.png)

![](https://cmd.dayi.ink/uploads/upload_424bb7c10a877e9da5efb53986f07388.png)

### 更换QWEN2
使用最新的Qwen2-7b大模型以提高对中文的支持能力。
![](https://cmd.dayi.ink/uploads/upload_9bba4110125a0935fcfd2100de1fc12b.png)
Qwen2在中文方面拥有比llama3更强的能力
![](https://cmd.dayi.ink/uploads/upload_fb64e1cc7e83a643244da0f692893346.png)
### Web UI
在docker上拉取WEB ui容器
![](https://cmd.dayi.ink/uploads/upload_b3afb46d18aed6e5ebabbfa9add3af90.png)
在浏览器中访问
![](https://cmd.dayi.ink/uploads/upload_0bdf3b3aa917e79b43e05f6be5128677.png)

![](https://cmd.dayi.ink/uploads/upload_b3ddcccd78a520c5d936227e21846ac5.png)
![](https://cmd.dayi.ink/uploads/upload_d8e8eb25f5ab2835bb626f71da432f2d.png)
### 调用API


