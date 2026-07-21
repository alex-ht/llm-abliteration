# llm-abliteration 程式說明文件

> 本文件詳細解釋此專案的設計、流程、核心演算法與各模組功能，方便後續維護、客製化或為新模型（如 google/gemma-4-E2B-it (the actual model the user downloaded via hf download) 系列）調適使用。

## 專案概述

**llm-abliteration** 是一個基於 Hugging Face `transformers` 的方向性行為編輯工具。核心概念不限於「拒絕/refusal」這一種行為：只要能準備一組「negative」（想減少的行為/風格）與「positive」（想保留或強化的行為/風格）prompt，就能測出一個方向向量，並選擇以下兩種模式之一套用：

- **Suppress 模式**（即原始的 Abliteration / 消融技術）：把方向向量從權重中永久移除，讓模型輸出較不會投影到該方向上。經典用例是移除「拒絕方向」（refusal direction），讓模型較不會對敏感/有害請求直接拒絕。
- **Boost 模式**：不動任何權重，改在推論時透過 forward hook 把方向向量加回 residual stream（activation steering／activation addition），讓輸出往 positive 方向偏移。

- 原始 suppress 概念來自論文與社群實作（andyrdt/refusal_direction、FailSpy/abliterator、Sumandora/remove-refusals-with-transformers 等）。
- 本實作特色：
  - 不依賴 TransformerLens。
  - 使用 Welford online mean + batched inference 加速測量。
  - 支援 **norm-preserving biprojected abliteration**（suppress 模式：保留權重 norm + 將方向對 positive 做正交投影）。
  - 記憶體效率高的 **sharded / single-file safetensors** 處理（大型模型只載入需要的 shard；小型模型直接支援單檔）。
  - 支援 Gemma 系列（含 tied weights 與多模態 vision 模型的特殊處理）。
  - YAML 驅動的多來源層 + 多目標層策略，並用 `mode: suppress` / `mode: boost` 切換兩種行為編輯模式。

已測試模型包含 Llama-3.2、Qwen2.5、Mistral 系列、gemma-3-12b/27b-it，以及較小的 Gemma 2B/4B 級模型。

**重要提醒**：Abliteration 無法 100% 保證完全解除審查，效果取決於資料集品質與選擇的層數/強度。使用後仍需自行評估安全與對齊風險。

## 核心概念

### 方向向量（Direction）

研究顯示，LLM 的許多行為（拒絕是最典型的例子）在 residual stream（殘差流）中常可被單一方向（或少數方向）所主導。

作法：
1. 準備兩組 prompt：
   - **negative**：想要減少/避開的行為或風格（e.g. 拒絕範例：如何偽造支票、如何駭入侵侵等）。
   - **positive**：想要保留/強化的行為或風格（e.g. 拒絕範例的對照：正常、無害的對話或問題）。
2. 對模型跑 forward（實際使用 `model.generate(..., output_hidden_states=True)` 取第一步生成時的 hidden states）。
3. 取每個 layer **最後一個 token**（pos=-1）的 activation，跨所有 prompt 計算 **平均向量**（negative_mean、positive_mean）。
4. `direction = negative_mean - positive_mean`（指向 negative 的方向）。
5. （可選）**Projected**：將 direction 對 positive_mean 做 Gram-Schmidt 正交化，移除其中 positive 方向的成分，讓方向更「乾淨」。

有了 direction 之後，可以選擇兩種方式套用它：

**Suppress 模式（消融）**：把模型特定權重矩陣（目前只動 **o_proj** 與 **mlp.down_proj**）的每一列（output neuron）減去它在 direction 上的投影成分：

- 標準版：`w -= scale * (w · d̂) * d̂`
- Norm-preserving 版：只對方向分量做消融，然後把原來的 norm 乘回來，盡量保留該 neuron 的輸出強度。

這是永久性的權重編輯，輸出是一個完整、獨立的新模型目錄。

**Boost 模式（activation steering）**：完全不動權重。因為 Llama/Gemma/Mistral/Qwen 這類架構的 `o_proj`/`mlp.down_proj` 都是 `nn.Linear(bias=False)`，線性層 `W @ x` 沒有 bias 項可以承載一個「固定偏移量」——換句話說，沒辦法把一個常數向量永久烤進權重矩陣裡。因此 boost 模式改採業界常見的 **activation addition**（如 ActAdd、Contrastive Activation Addition／CAA）做法：在推論時對目標層的輸出（residual stream）用 forward hook 動態加上 `scale * (-direction 的單位向量)`（即指向 positive 的單位向量）。這個效果只在安裝了 hook 之後才存在——`sharded_ablate.py` 的 boost 模式輸出目錄裡的權重跟原模型一模一樣，只是多了一個 `steering_vectors.pt` side-car 檔；用純 `transformers.from_pretrained()` 載入不會有任何效果，必須透過 `chat.py`（或 `utils/steering.py`）安裝 hook 才會生效。

### 為什麼只修改 o_proj 與 down_proj？（suppress 模式）

- `o_proj`：attention 輸出投影，把 attention 結果寫回 residual stream。
- `mlp.down_proj`：在 Llama/Gemma 類架構中，MLP 的 gate/up 後的 down 投影，把 FFN 結果寫回 residual。
- 這兩個位置相當於「resid_post」貢獻的主要寫入點，消融它們對 residual stream 的影響，效果類似傳統 TransformerLens 針對 resid_post 的作法。
- 只改這兩個可大幅減少計算量與對模型其他能力的破壞。
- Boost 模式則是直接對**整個 decoder layer 的輸出**（也就是 resid_post 本身）加 hook，不特別區分 o_proj/down_proj 的個別貢獻。

### 為什麼用 generate 而不是直接 model(...)？

使用 `generate(..., output_hidden_states=True, max_new_tokens=1)` 可以在「處理完 prompt 後、產生第一個新 token 之前」取得對應的 hidden states，實務上等價於取得 prompt 最後一個 token 的 layer hidden state（許多 abliteration 實作都採用此 hack）。

## 完整工作流程

1. **準備 / 確認資料集**（`data/negative.parquet`、`data/positive.parquet` 或自訂）。
2. **測量（Measure）**：`python measure.py -m <model> -o <measurements.pt> [--projected]`
   - 計算所有 layer 的 negative_mean / positive_mean / direction。
   - 輸出 torch.save 的 dict。
3. **分析（Analyze）**：`python analyze.py <measurements.pt> -c`
   - 印出每層的 cosine similarity、norm、SNR、purity、估計 signal quality。
   - `-c` 會畫圖（`direction_analysis.png`），通常「中層到中後層」是方向訊號最強、最適合取用的來源。
4. **撰寫 YAML 策略檔**（參考 `gemma3-12b-it.yml` 為 suppress 模式範例，`gemma-4-E2B-it-boost.yml` 為 boost 模式範例），並用 `mode: suppress`（預設）或 `mode: boost` 選擇模式。
5. **執行編輯**：`python sharded_ablate.py your.yml [--normpreserve] [--projected]`
   - `mode: suppress`：產生一個新的 model 目錄，裡面只有被改過的 shard + 完整 config/tokenizer 檔。`--normpreserve` 只在此模式有效。
   - `mode: boost`：產生一個新的 model 目錄，權重與原模型完全相同，外加一個 `steering_vectors.pt`。`--normpreserve` 在此模式無效果（會印警告並忽略）。
6. **測試**：
   - `python chat.py -m <output_dir>` — 若目錄含 `steering_vectors.pt` 會自動載入並安裝 hook；也可用 `--steering-vectors <path>` 指定其他 side-car、`--steer-scale <float>` 整體縮放。
   - 或用 transformers 直接 load 測試 negative/positive prompt（注意：boost 模式必須透過 `chat.py` 或 `utils/steering.py` 安裝 hook 才會生效，純 `from_pretrained()` 不會有效果）。
   - 建議與原始模型做定性/定量比較（compare.py 為舊版工具，可自行擴充）。

## 各檔案與模組詳細說明

### 主要入口腳本

| 檔案                  | 用途                                   | 關鍵參數 / 行為 |
|-----------------------|----------------------------------------|-----------------|
| `measure.py`          | 計算各層 direction（negative_mean/positive_mean/direction） | `-m` 模型, `-o` 輸出 .pt, `--projected`, `--batch-size`, `--clip`, `--deccp`, `--quant-measure 4bit/8bit`, `--data-negative/positive` |
| `analyze.py`          | 統計 + 可選繪圖，幫助決定 YAML 策略    | `data_file`, `-c/--chart` |
| `sharded_ablate.py`   | 依 YAML 執行 suppress（消融權重）或 boost（寫出 steering_vectors.pt） | `your.yml`（含 `mode: suppress\|boost`）, `--normpreserve`（僅 suppress）, `--projected` |
| `chat.py`             | 簡單聊天測試；若模型目錄含 `steering_vectors.pt` 會自動安裝 hook | `-m`, `--steering-vectors`, `--steer-scale`, 各種 quant / precision |
| `compare.py`          | 比較兩個模型權重差異（繼承舊碼）       | `-a` `-b` |
| `bnbquant.py`         | 把完整權重模型轉 4bit/8bit 存檔（僅供測量用） | 位置參數 model output quant |

### utils/ 工具模組

- **`utils/data.py`**：`load_data(path)` 統一支援 `.txt`（每行一 prompt）、`.parquet`（text 欄）、`.json`（list）、`.jsonl`（{"text": ...}）。
- **`utils/models.py`**：`has_tied_weights(model_type)` — 偵測 Gemma 家族（gemma / gemma2 / gemma3 / paligemma），載入後需呼叫 `model.tie_weights()`。
- **`utils/clip.py`**：`magnitude_clip` — 對 activation 做對稱 Winsorize（限幅），測量時可指定 `--clip 0.99` 減少極端值影響。
- **`utils/steering.py`**（新）：boost 模式的執行端。`load_steering_vectors(path)` 讀取 `steering_vectors.pt`；`get_decoder_layers(model)` 用跟 `measure.py` 相同的 `model.model` → `.language_model` fallback 找到 decoder layer 清單；`install_steering_hooks(model, entries, global_scale)` 對每個目標層註冊 forward hook，把 `scale * global_scale * vector` 加到該層輸出（同時處理回傳值是 tuple 或裸 tensor 的情況，涵蓋 prefill 與逐 token decode 的所有 forward call）；`remove_steering_hooks(handles)` 反安裝。
- **`utils/score.py`**：舊版 scoring / analyze_direction 工具（用於計算 activation 在方向向量上的投影分數），目前主要由 compare / 舊流程使用。
- **`utils/sparsify.py`**：提供多種 sparsify 方法（magnitude、percentile、topk、soft threshold 等），但目前 `sharded_ablate.py` 內建簡易版 `magnitude_sparsify`（依元素數量比例取 top-k），suppress 與 boost 模式共用同一份實作。

### 資料轉換工具

- `parquet_to_jsonl.py` / `jsonl_to_parquet.py`：方便自訂資料集時在兩種格式間轉換（大型資料建議用 parquet 較省空間）。

### 設定檔範例

- `gemma3-12b-it.yml`（suppress 模式，`mode` 省略即預設 `suppress`）：
  ```yaml
  model: google/gemma-3-12b-it
  measurements: outedg3e.refuse   # measure.py 輸出的 .pt 檔
  output: outg3c4                 # 輸出目錄
  ablate:
    - layer: 11
      measurement: 23
      scale: 1.0
      sparsity: 0.00
    ...
  ```
  - `layer`：目標層（suppress 模式會改該層的 o_proj 與 down_proj；boost 模式則是安裝 hook 的 decoder layer）。
  - `measurement`：使用哪一層算出來的 direction（可與 layer 不同，允許多來源策略）。
  - `scale`：suppress 模式為消融強度（1.0 為標準，代表移除接近 100% 的投影分量）；boost 模式為加到單位向量上的**絕對幅度**，兩者單位完全不同，boost 模式請參考 `sharded_ablate.py` 執行時印出的該層 positive-mean norm 作為量級參考。
  - `sparsity`：>0 時只保留 direction 中 magnitude 最大的前 `sparsity` 比例元素（注意：這裡的命名是「保留比例」，不是傳統 sparsity 0.9=只留10%）。

- `gemma-4-E2B-it-boost.yml`（boost 模式範例）：與上面結構相同，只多了一行 `mode: boost`。執行後輸出目錄的權重與原模型完全一致，額外多一個 `steering_vectors.pt`；`--normpreserve` 在此模式下無效果（會印警告並忽略）。

## 記憶體與效能考量

- **測量階段**：最吃 VRAM。建議：
  - 小模型直接全精度。
  - 大模型先用 `bnbquant.py` 轉 4bit 存檔，再用 `--quant-measure 4bit` 載入測量（或讓 script 自動偵測 quantization_config）。
  - 調低 `--batch-size`（預設 32，視 VRAM 調整為 8/16/64）。
  - `--clip 0.99` 可稍微降低峰值。
- **消融階段**：sharded 版本只載入需要改的 shard，峰值記憶體低很多。單檔小模型會載入整個 state_dict（2B 級別通常 5-10GB RAM 臨時使用，可接受）。
- 消融輸出永遠是**完整權重**（bf16/fp16 視原始模型），之後可再自行 quant。
- CUDA 為假設環境（device_map="auto"）。

## 進階選項與變體

- **測量時 `--projected`**：在計算 direction 時就對同層 positive_mean 做正交。
- **suppress 模式 `--projected`**：在套用前，再用**目標層**的 positive_mean 對 direction 做一次正交（即使來源 measurement 層不同）。
- **boost 模式 `--projected`**：方向相反——用**目標層**的 negative_mean 對 boost 方向做正交，避免加強時意外往 negative 方向偏移。
- **suppress 模式 `--normpreserve`**：使用 norm-preserving 版本，只消融方向、不改變各 output neuron 的權重 magnitude。boost 模式沒有「權重 norm」這個概念，此旗標會被忽略並印警告。
- **YAML 中 `sparsity`**：實驗性，兩種模式都適用，只用 direction 的稀疏子集。
- 多來源層策略：可讓某些 layer 用較早的 measurement，某些用較晚的，實務上常見「measurement 取 23/29，消融 11~41 層」這種分段作法（見 gemma3 範例），suppress/boost 皆可套用同一套策略思路。
- **組合兩種模式**：因為 boost 模式輸出目錄的權重跟原模型相同，你可以把 suppress 模式產出的模型目錄，搭配另外跑出來的 boost 模式 `steering_vectors.pt`，用 `python chat.py -m <suppress輸出目錄> --steering-vectors <boost輸出目錄>/steering_vectors.pt` 同時套用兩種效果。

## 為 Gemma 系列模型的特殊處理

- `has_tied_weights`：Gemma 系列 embedding 與 lm_head 權重 tied，載入後需 tie。
- Vision 模型（gemma-3 具備 vision_config 的 it 版本）：measure.py 會自動切換 `AutoModelForImageTextToText` + `AutoProcessor`，並用 processor 做 text-only chat template。
- 目前消融只針對 text 部分的 language model layers（sharded 偵測 prefix 時會抓到正確的 "model" 或 "language_model" 前綴）。

## 如何讓 google/gemma-*-*b-it（小型模型）也能跑

Gemma-4 系列小型模型（如 `google/gemma-4-E2B-it`）通常為單一 `model.safetensors`（加上 vision/audio tower），且容易 fit 進記憶體。我們只對 language_model 部分的 layers 進行消融。

### 推薦執行步驟

1. **環境準備**（同 README）
   ```bash
   pip install -r requirements.txt
   ```

2. **執行測量**（不需 quant）
   ```bash
   python measure.py \
     -m google/gemma-4-E2B-it \
     -o gemma4-e2b.direction \
     --batch-size 16
   ```
   - 若要用 projected 變體：加上 `--projected`。
   - 對中文模型可加 `--deccp`。
   - 自訂資料：`--data-negative your-negative.parquet --data-positive your-positive.parquet`

3. **分析結果，決定策略**
   ```bash
   python analyze.py gemma4-e2b.direction -c
   ```
   - 看 console 輸出與產生的 `direction_analysis.png`。
   - 挑選 signal quality 高、purity 好、cosine sim 適中的中層作為 measurement 來源。
   - 目標編輯層通常從中層開始往後（例如 8~18 之類，視總層數）。

4. **撰寫 YAML**（範例，依 analyze 結果調整數字；`mode: suppress` 可省略，`mode: boost` 必須明寫）
   ```yaml
   model: google/gemma-4-E2B-it
   measurements: gemma4-e2b.direction
   output: gemma4-e2b-abliterated
   ablate:
     - layer: 8
       measurement: 12
       scale: 1.0
       sparsity: 0.0
     - layer: 10
       measurement: 12
       scale: 1.0
       sparsity: 0.0
     - layer: 15
       measurement: 18
       scale: 1.0
       sparsity: 0.0
     # 總共 35 層，建議先跑 analyze.py 再依圖表調整 measurement 來源層與目標層
     # ... 繼續加中後層（例如 12~28 左右）
   ```
   若要改用 boost 模式，加一行 `mode: boost`，並把 `scale` 從「消融強度」改成「絕對加成幅度」（參考 `gemma-4-E2B-it-boost.yml`）。

5. **執行編輯**（現在單檔模型已直接支援）
   ```bash
   python sharded_ablate.py gemma4-e2b-ablit.yml --normpreserve --projected
   ```
   - `mode: suppress`：輸出目錄 `gemma4-e2b-abliterated/` 即可直接當 HF 模型使用。
   - `mode: boost`：輸出目錄權重跟原模型一樣，多一個 `steering_vectors.pt`，須搭配 `chat.py` 才會生效；`--normpreserve` 會被忽略。

6. **驗證**
   ```bash
   python chat.py -m gemma4-e2b-abliterated
   ```
   或
   ```python
   from transformers import AutoModelForCausalLM, AutoTokenizer
   model = AutoModelForImageTextToText.from_pretrained("gemma4-e2b-abliterated", device_map="auto")
   # ... 測試 prompt
   ```
   （注意：boost 模式輸出必須透過 `chat.py` 或 `utils/steering.py` 安裝 hook，上面這段純 `from_pretrained()` 範例只適用 suppress 模式輸出。）

### 注意事項（小型模型）

- Gemma 2B/4B 總層數少（Gemma-2-2B 是 26 層？Gemma-3-4B 更多），analyze 時觀察範圍會比較集中。
- Suppress 模式建議先用 `--normpreserve` + `--projected` 組合，效果通常較穩定。
- 消融後模型仍為完整精度，可再用 bitsandbytes 或其他 quant 工具壓縮。
- 若想用原始的（未改動的） positive/negative 資料集以外的資料，強烈建議先小規模實驗不同 measurement 來源層的影響。

## 常見問題與提示

- **載入 Gemma 模型報 tied weights 相關錯**：measure.py 已自動處理 `has_tied_weights` 後呼叫 tie_weights。
- **sharded_ablate 找不到 index.json**：已修復，現在會自動退回單檔模式。
- **消融後模型還是會拒絕（suppress 模式）**：可能 measurement 來源層選得不好、scale 不夠、或資料集不夠對比強烈。試試更多層、調高 scale、或自訂更 sharp 的 negative/positive 集。
- **boost 模式套用後感覺沒效果**：檢查是不是用純 `transformers.from_pretrained()` 載入（不會套用 steering）；確認 `chat.py` 啟動時有印出「Loaded N steering vector(s)」；`scale` 可能太小——參考印出的 `ref_norm` 調高。
- **boost 模式套用後輸出崩壞/離題**：`scale` 通常太大，或選的層太多/太前面；降低 scale 或減少目標層。
- **記憶體不足**：測量階段優先用 4bit 測量；suppress 模式單檔小模型通常沒問題；boost 模式因為不用把整個 state_dict 讀進來改寫，記憶體需求比 suppress 更低。
- **想只消融特定子集合**：目前硬寫只動 o_proj + down_proj，如需擴充到其他模組（q_proj、up_proj...）需修改 `sharded_ablate.py` 內的 pattern 與 marching_orders 處理。
- **中文模型**：使用 `--deccp` 會從 augmxnt/deccp 資料集額外加入「被審查話題」當 negative。
- **舊版測量檔（.pt）無法使用**：舊版 key 為 `harmful_*`/`harmless_*`/`refuse_*`，本版改為 `negative_*`/`positive_*`/`direction_*`，需重新執行 `measure.py`。

## 參考資料與致謝

- 原始論文/技術說明：https://huggingface.co/blog/mlabonne/abliteration
- Norm-preserving biprojected 變體：https://huggingface.co/blog/grimjim/norm-preserving-biprojected-abliteration
- 相關專案：Orion-zhen/abliteration、FailSpy/abliterator、Sumandora/remove-refusals-with-transformers、AUGMXNT/deccp
- 本專案在這些基礎上做了工程優化（sharded 處理、Welford batched、norm preserve、Gemma 支援、單檔模型相容等）。

---

如需進一步客製（例如支援更多模組消融、自動挑選最佳層的 script、對特定模型的 layer 命名適配），歡迎提出需求或直接修改對應的 pattern 與 YAML 產生邏輯。

祝消融順利，產生出你想要的「不說不」的模型！
