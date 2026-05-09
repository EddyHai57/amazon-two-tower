# Amazon All_Beauty 数据理解笔记

## 1. 数据来源与当前配置

当前配置来自 `configs/amazon_all_beauty_phase1.yaml`：

- `dataset_name`：`McAuley-Lab/Amazon-Reviews-2023`
  - 含义：HuggingFace 上的 Amazon Reviews 2023 数据集名称。
  - 当前用途：inspection 脚本用它定位要加载的数据集。
- `review_config`：`raw_review_All_Beauty`
  - 含义：All_Beauty 类目的原始评论/评分数据配置。
  - 当前用途：inspection 脚本用它加载 reviews 用户行为表。
- `meta_config`：`raw_meta_All_Beauty`
  - 含义：All_Beauty 类目的商品元数据配置。
  - 当前用途：inspection 脚本用它加载 meta 物品信息表。
- `inspection_sample_size`：`100000`
  - 含义：如果 full load 失败并进入 streaming fallback，只检查前 100000 行。
  - 当前用途：本次实际使用了 `full_load`，所以没有用它限制样本。
- `min_user_interactions`：`3`
  - 含义：后续预处理时可能用于过滤交互太少的用户。
  - 当前用途：占位参数，inspection 阶段没有使用。
- `min_item_interactions`：`3`
  - 含义：后续预处理时可能用于过滤交互太少的物品。
  - 当前用途：占位参数，inspection 阶段没有使用。
- `positive_rating_threshold`：`4`
  - 含义：后续可能把 `rating >= 4` 的行为视为正反馈。
  - 当前用途：占位参数，inspection 阶段没有使用。

当前已经用于 inspection 的参数是：

- `dataset_name`
- `review_config`
- `meta_config`
- `inspection_sample_size` 仅作为 streaming fallback 的备用限制，本次 `full_load` 成功后没有实际限制数据量。

当前只是后续 preprocessing 占位的参数是：

- `min_user_interactions`
- `min_item_interactions`
- `positive_rating_threshold`

项目把业务参数放在 config 中，而不是硬编码在脚本里，是为了保证实验可复现、方便记录每次实验条件，并且避免后续改参数时悄悄破坏 ItemCF 和双塔模型之间的可比性。

## 2. 数据集整体概览

本次数据检查结果来自 `outputs/inspection_all_beauty.md`：

- loading strategy used：`full_load`
- review row count：701528
- meta row count：112590
- unique user_id count：631986
- unique parent_asin count：112565

这些数字说明：

- `review row count` 表示用户评论/评分行为数量。All_Beauty 当前有 701528 条用户行为记录。
- `meta row count` 表示商品元数据数量。All_Beauty 当前有 112590 条商品元数据记录。
- `unique user_id count` 表示出现过的唯一用户数。631986 个唯一用户接近 701528 条行为，说明很多用户只有很少行为，数据非常稀疏。
- `unique parent_asin count` 表示 reviews 中出现过的唯一父商品数。112565 个唯一 `parent_asin` 接近 112590 条 meta 记录，说明多数商品在元数据中有独立的 `parent_asin`。

对推荐召回来说，这意味着数据可以用于“用户-物品交互”建模，但在训练前必须认真做交互过滤和时间切分，否则很多用户或物品的数据太少，会影响训练和评估稳定性。

## 3. 两张核心“表”的理解

可以先把数据理解成两张表：

- reviews 用户行为表：记录用户对商品的评分和评论行为。
- meta 物品信息表：记录商品自身的标题、描述、类目、图片等元数据。

### 3.1 reviews 用户行为表

reviews 表代表用户对商品的评分/评论行为，是后续构造“用户-物品交互”的主要来源。

#### rating

- 字段名：`rating`
- 中文含义：用户给商品的评分。
- 样例中大概长什么样：`5.0`、`4.0`。
- 对推荐召回项目的作用：可以用来构造隐式反馈，例如后续可能把 `rating >= 4` 当作正样本。
- Phase 1 是否使用：使用。它是构造正反馈的关键字段，但具体阈值仍以 Eddy 确认为准。

#### title

- 字段名：`title`
- 中文含义：用户评论标题。
- 样例中大概长什么样：一小段用户写的评论标题，例如对商品体验的简短总结。
- 对推荐召回项目的作用：它反映用户评论内容，不是商品自身内容；可用于后续评论文本分析，但不适合作为 Phase 1 的核心 item 文本。
- Phase 1 是否使用：暂时不使用。

#### text

- 字段名：`text`
- 中文含义：用户评论正文。
- 样例中大概长什么样：用户写的一段评论文本，描述使用感受、气味、效果等。
- 对推荐召回项目的作用：它是用户反馈文本，不是商品官方描述。后续如果研究 review-aware 推荐可以考虑，但 Phase 1 不用它做 item embedding。
- Phase 1 是否使用：暂时不使用。

#### images

- 字段名：`images`
- 中文含义：用户评论中附带的图片。
- 样例中大概长什么样：空列表 `[]`，或图片相关结构。
- 对推荐召回项目的作用：属于用户评论侧的多模态信息，处理成本较高。
- Phase 1 是否使用：暂时不使用。

#### asin

- 字段名：`asin`
- 中文含义：Amazon 商品或商品变体 ID。
- 样例中大概长什么样：`B00YQ6X8EO`、`B081TJ8YS3`。
- 对推荐召回项目的作用：可以定位具体商品变体，但同一父商品下可能有多个变体。
- Phase 1 是否使用：暂时不作为主要 item_id，优先使用 `parent_asin`。

#### parent_asin

- 字段名：`parent_asin`
- 中文含义：父商品 ID，更接近推荐召回中的商品粒度。
- 样例中大概长什么样：`B00YQ6X8EO`、`B097R46CSY`。
- 对推荐召回项目的作用：用于表示被召回的 item，可以把同一商品的不同变体行为聚合到父商品上。
- Phase 1 是否使用：使用。暂定作为 item_id，但最终预处理规则仍以 Eddy 确认为准。

#### user_id

- 字段名：`user_id`
- 中文含义：用户原始 ID。
- 样例中大概长什么样：一串匿名用户标识，例如 `AGKHLEW2SOWHNMFQIJGBECAF7INQ`。
- 对推荐召回项目的作用：用于表示用户，是构造用户历史行为、训练双塔 user tower 的基础。
- Phase 1 是否使用：使用。

#### timestamp

- 字段名：`timestamp`
- 中文含义：用户行为发生时间。
- 样例中大概长什么样：毫秒级时间戳，例如 `1588687728923`。
- 对推荐召回项目的作用：用于按时间切分 train/valid/test，保证 test 行为严格晚于 train+valid，避免数据泄露。
- Phase 1 是否使用：使用。

#### helpful_vote

- 字段名：`helpful_vote`
- 中文含义：该评论被其他用户认为有帮助的票数。
- 样例中大概长什么样：`0`、`1`、`2`。
- 对推荐召回项目的作用：更像评论质量或可信度信号，不是最小交互建模必需字段。
- Phase 1 是否使用：暂时不使用。

#### verified_purchase

- 字段名：`verified_purchase`
- 中文含义：是否为 Amazon 认证购买后的评论。
- 样例中大概长什么样：`true`。
- 对推荐召回项目的作用：可用于后续过滤或分析行为可信度，但会影响样本规模和实验可比性。
- Phase 1 是否使用：暂时不使用，除非 Eddy 后续确认过滤策略。

### 3.2 meta 物品信息表

meta 表代表商品本身的元数据，是后续构造 item 特征，特别是文本 embedding 的主要来源。

#### main_category

- 字段名：`main_category`
- 中文含义：商品主类目。
- 样例中大概长什么样：`All Beauty`。
- 对推荐召回项目的作用：可作为类目信息，帮助分析商品分布，也可作为后续 item 特征。
- Phase 1 是否使用：可用于分析，最小 ID-only 基线不依赖它。

#### title

- 字段名：`title`
- 中文含义：商品标题。
- 样例中大概长什么样：商品名、规格、品牌和包装信息组成的一段标题。
- 对推荐召回项目的作用：这是 item 文本 embedding 的优先文本来源之一。
- Phase 1 是否使用：Phase 1 的 ID-only 训练暂时不用；后续文本 embedding 阶段优先使用。

#### average_rating

- 字段名：`average_rating`
- 中文含义：商品平均评分。
- 样例中大概长什么样：`4.8`、`4.5`、`4.4`。
- 对推荐召回项目的作用：可用于商品质量分析，但它是聚合统计，使用时要小心时间泄露。
- Phase 1 是否使用：暂时不使用。

#### rating_number

- 字段名：`rating_number`
- 中文含义：商品评分数量。
- 样例中大概长什么样：`10`、`3`、`26`。
- 对推荐召回项目的作用：可表示商品热度，但也是聚合统计，评估时需要谨慎。
- Phase 1 是否使用：暂时不使用。

#### features

- 字段名：`features`
- 中文含义：商品卖点或特征列表。
- 样例中大概长什么样：空列表 `[]`，或多条商品特征文本。
- 对推荐召回项目的作用：后续可与 title、description 一起作为 item 文本。
- Phase 1 是否使用：暂时不使用。

#### description

- 字段名：`description`
- 中文含义：商品描述。
- 样例中大概长什么样：空列表 `[]`，或商品介绍文本列表。
- 对推荐召回项目的作用：这是 item 文本 embedding 的优先文本来源之一，通常可与 `meta.title` 拼接。
- Phase 1 是否使用：Phase 1 的 ID-only 训练暂时不用；后续文本 embedding 阶段优先使用。

#### price

- 字段名：`price`
- 中文含义：商品价格。
- 样例中大概长什么样：`None` 或价格字符串。
- 对推荐召回项目的作用：可作为后续分析或特征，但可能缺失、格式不统一。
- Phase 1 是否使用：暂时不使用。

#### images

- 字段名：`images`
- 中文含义：商品图片信息。
- 样例中大概长什么样：包含 `hi_res`、`large`、`thumb`、`variant` 等图片 URL 列表。
- 对推荐召回项目的作用：可用于多模态推荐，但需要图像处理或图像 embedding。
- Phase 1 是否使用：暂时不使用。

#### videos

- 字段名：`videos`
- 中文含义：商品视频信息。
- 样例中大概长什么样：包含 `title`、`url`、`user_id` 等列表，样例中多为空。
- 对推荐召回项目的作用：属于多模态扩展，处理复杂度较高。
- Phase 1 是否使用：暂时不使用。

#### store

- 字段名：`store`
- 中文含义：商品店铺或品牌来源。
- 样例中大概长什么样：`Howard Products`、`Yes To`。
- 对推荐召回项目的作用：可作为品牌/店铺侧信息，后续可用于分析或特征。
- Phase 1 是否使用：暂时不使用。

#### categories

- 字段名：`categories`
- 中文含义：商品类目路径或类目列表。
- 样例中大概长什么样：空列表 `[]`，或多级类目列表。
- 对推荐召回项目的作用：可作为物品类目特征或后续分析字段。
- Phase 1 是否使用：可用于分析，最小 ID-only 基线不依赖它。

#### details

- 字段名：`details`
- 中文含义：商品详情字典，常包含尺寸、品牌、UPC、制造商等信息。
- 样例中大概长什么样：JSON 字符串，内部字段较杂。
- 对推荐召回项目的作用：信息丰富但结构复杂，清洗成本高。
- Phase 1 是否使用：暂时不使用。

#### parent_asin

- 字段名：`parent_asin`
- 中文含义：父商品 ID。
- 样例中大概长什么样：`B01CUPMQZE`、`B076WQZGPM`。
- 对推荐召回项目的作用：用于把 reviews 行为表和 meta 物品信息表连接起来。
- Phase 1 是否使用：使用。它是 item 维度的关键连接字段。

#### bought_together

- 字段名：`bought_together`
- 中文含义：经常一起购买的商品信息。
- 样例中大概长什么样：`null`。
- 对推荐召回项目的作用：可能带有共购关系信息，但不属于最小交互建模。
- Phase 1 是否使用：暂时不使用。

#### subtitle

- 字段名：`subtitle`
- 中文含义：商品副标题。
- 样例中大概长什么样：`null` 或简短文本。
- 对推荐召回项目的作用：可补充商品文本，但缺失可能较多。
- Phase 1 是否使用：暂时不使用。

#### author

- 字段名：`author`
- 中文含义：作者字段，部分商品可能有值。
- 样例中大概长什么样：`null`。
- 对推荐召回项目的作用：All_Beauty 类目中通常不是核心字段。
- Phase 1 是否使用：暂时不使用。

## 4. Phase 1 最小建模字段

reviews 中最小可用字段：

- `user_id`：表示用户，用于构造用户历史和用户侧 ID。
- `parent_asin`：表示召回物品，用于构造 item_id。
- `rating`：用于构造隐式反馈，例如后续可能使用 `rating >= 4`。
- `timestamp`：用于时间切分，避免数据泄露。

meta 中最小可用字段：

- `parent_asin`：用于和 reviews 表对齐物品。
- `title`：商品标题，后续用于生成文本 embedding。
- `description`：商品描述，后续可与 `title` 拼接生成文本 embedding。
- `categories` 或 `main_category`：物品类目信息，可用于分析或后续特征。

Phase 1 的最小训练闭环应先围绕 `user_id`、`parent_asin`、`rating`、`timestamp` 完成数据切分和 ID-only baseline。`title`、`description`、`categories`、`main_category` 可以先作为理解数据和后续扩展的准备字段。

## 5. Phase 1 暂时不用的字段

reviews 中暂时不用：

- `images`：评论图片，属于多模态扩展。
- `helpful_vote`：评论有用票数，偏评论质量信号，不是最小交互字段。
- `verified_purchase`：认证购买标记，可能用于过滤，但会影响样本规模和实验可比性。
- `text` 评论正文：这是用户评论内容，不是商品自身内容。
- `title` 评论标题：这是用户评论标题，不是商品自身标题。

meta 中暂时不用：

- `images`：商品图片，属于多模态扩展，不属于 Phase 1。
- `videos`：商品视频，属于多模态扩展，不属于 Phase 1。
- `bought_together`：共购信息，可能有用，但会改变召回信号来源。
- `subtitle`：商品副标题，样例中为 `null`，暂时不是核心字段。
- `author`：All_Beauty 类目中通常不是核心字段。
- `details`：结构复杂，清洗成本高。
- `price`：可能缺失或格式不统一，暂时不作为核心特征。

特别注意：

- review text 和 review title 是用户评论内容，不是商品自身内容。
- item 文本 embedding 应优先使用 meta 表里的 `title` 和 `description`。
- `price`、`details` 等字段可能有缺失或结构复杂，Phase 1 暂时不作为核心特征。
- `images`、`videos` 属于多模态扩展，不属于 Phase 1。

## 6. parent_asin 与 asin 的区别

- `asin` 更像具体变体 ID，同款商品不同规格、包装或版本可能有不同 `asin`。
- `parent_asin` 更像父商品 ID，更适合召回阶段作为 item 粒度。
- 使用 `parent_asin` 可以减少物品过度拆分，让同一商品的行为聚合到一起。
- Phase 1 暂定使用 `parent_asin` 作为 item_id，但最终预处理规则仍以 Eddy 确认为准。

一个直观例子是：reviews 样例中有一条记录的 `asin` 是 `B07PNNCSP9`，但 `parent_asin` 是 `B097R46CSY`。这说明具体变体和父商品可能不是同一个 ID。

## 7. 为什么现在还不能直接训练模型

虽然 Amazon All_Beauty 数据已经成功加载，但现在还不能直接训练模型，因为还缺少：

- `rating` 分布分析。
- `rating >= 4` 后的正样本规模。
- 用户交互数分布。
- item 交互数分布。
- k-core 过滤后剩余规模。
- train/valid/test 时间切分结果。
- ItemCF baseline 的输入格式。
- ID-only 双塔的输入格式。

这些需要下一步通过 `analyze_interactions.py` 或等价分析完成。只有先知道过滤后还剩多少用户、物品和交互，才能安全决定预处理策略。

## 8. 当前已知环境问题

之前 HuggingFace datasets 加载失败，失败信息是：

```text
Dataset scripts are no longer supported, but found Amazon-Reviews-2023.py
```

当前解决方式：

- 在项目独立 `.venv` 中安装兼容版本依赖。
- 当前可用依赖版本：
  - `datasets==2.17.0`
  - `huggingface_hub==0.36.2`

后续运行项目脚本时应进入：

```bash
cd /workspace/amazon-two-tower
source .venv/bin/activate
```

然后再运行项目脚本。

## 9. 当前数据理解结论

- Amazon All_Beauty 数据已经可以成功加载。
- review 数据和 meta 数据都可用。
- 当前数据非常稀疏，因为 unique user_id 接近 review row count；进一步分析发现 93.28% 的用户只有 1 条正向交互。
- `rating>=4` 正样本有 500107 条，但 `user>=3,item>=3` k-core 后只剩 8657 条 interaction，说明 All_Beauty 不适合作为 Phase 1 简历主实验数据集。
- All_Beauty 保留为 Phase 0 工程验证数据集，用于验证数据加载、inspection、interaction analysis 和日志流程。
- Phase 1 需要切换到更合适的大类目，但不直接锁 Electronics，需要先比较 Electronics、Video_Games、Movies_and_TV。
- 当前还没有生成 train/valid/test、ID mapping、ItemCF 输出或模型 checkpoint。
