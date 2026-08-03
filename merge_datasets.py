"""
既然您決定讓系統透過 AI 直接辨識違規（端到端 End-to-End 模型），那麼模型的訓練難度會提高。因為 YOLO 必須在「單張靜態圖片」中，同時辨識出紅燈、停止線、車輛位置三者的複合空間關係，並直接輸出 red-light-violation（闖紅燈違規）。
為了達成這個目標，第 5 個資料集的標註與融合方式必須進行非常嚴格的調整，否則模型一定會產生严重的誤判（例如綠燈時車輛過線也被當作違規）。
## 📐 關鍵：AI 直接辨識的標註與數據篩選原則

   1. 嚴格的違規標籤定義：
   在第 5 個資料集中，red-light-violation 標籤框出的範圍，必須是「正在超越停止線的那輛違規車子」。
   2. 畫面必須包含紅燈：
   被標註為 red-light-violation 的圖片中，畫面上一定要拍到亮起的紅燈。如果畫面沒拍到紅燈（只有車和線），AI 就無法學習到「紅燈」這個關鍵觸發特徵，請務必剔除沒拍到紅燈的違規照片。
   3. 必須大量引入「綠燈過線」的背景圖（Negative Samples）：
   您的第 1 到第 3 個資料集（包含綠燈、車輛、機車）在此時扮演極度重要的角色！當這些資料集融合進來後，因為它們有車子正常行駛、有綠燈，但沒有 red-light-violation 標籤，這會強迫 AI 學習到：「只有當紅燈亮起且車子壓線時才要框出違規；綠燈壓線時什麼都不用框」。

------------------------------
## 📂 第 5 個資料集的類別映射設定（假設）
假設您第 5 個資料集原本的 data.yaml 定義如下（請依據實況調整）：

* 0: car（普通汽車）
* 1: stop-line-violation（正在闖紅燈的違規車輛/行為）

我們要將 0 映射到標準類別 3 (car)，將 1 映射到標準類別 9 (red-light-violation)。
------------------------------
## 🛠️ 專為「AI 直接辨識違規」優化的五合一融合腳本
以下腳本已將類別精簡為 10 大類，並針對 AI 直接偵測違規的邏輯進行了優化：

## 💡 接下來建議的開發步驟
為了讓 AI 直接辨識違規的準確度最高，建議您執行以下確認：

* 請檢查第 5 個資料集原本的 data.yaml 內容，確認代表「闖紅燈違規」的舊編號到底是幾號？（以便修改腳本第 31~34 行的對應關係）。
* 您是否需要我提供如何在 YOLOv8 訓練設定中調整參數（例如針對特定類別加權），來提升這種複合特徵（紅燈+線+車）的偵測敏感度？

"""

import os
import shutil

# ==================== 1. 設定區 ====================
DATASETS = [
    "traffic-light-taiwan.v1i.yolov8",
    "Traffic-Violation-Detection.v4i.yolov8",
    "3_project.v5i.yolov8",
    "double-white-lines.dataset",
    "stop-line-violation.dataset"  # 👈 請改為您第 5 個資料集的實際資料夾名稱
]

# 類別對照表 {舊編號: 新標準編號}
DATASET_MAPPINGS = [
    # 第 1 個資料集: 綠(0)、紅(1)、黃(2)
    {0: 0, 1: 1, 2: 2},

    # 第 2 個資料集: 違規細分標籤合併到 motor(4) 與 bike(5)
    {0: 4, 1: 4, 2: 5, 3: 5, 4: 4, 5: 4, 6: 4, 7: 5, 8: 5},

    # 第 3 個資料集: 自行車(5)、汽車(3)、機車(4)、行人(6)、卡車(7)
    {0: 5, 1: 3, 2: 4, 3: 6, 4: 7},

    # 第 4 個資料集: 雙白實線(8)、汽車(3)
    {0: 8, 1: 3},

    # 第 5 個資料集：超過紅燈停止線數據集
    # ⚠️ 請對照該資料集原本的 data.yaml，確認哪一個編號是代表「違規車輛」
    {
        0: 3,  # 假設舊編號 0 是正常汽車 -> 轉為新汽車(3)
        1: 9  # 假設舊編號 1 是闖紅燈違規車輛 -> 轉為新闖紅燈違規(9)
    }
]

OUTPUT_DIR = "merged_dataset"
SPLITS = ["train", "val", "test"]

# ==================== 2. 初始化目標資料夾 ====================
for split in SPLITS:
    os.makedirs(os.path.join(OUTPUT_DIR, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, split, "labels"), exist_ok=True)

print("🚀 開始融合五個資料集（AI 直辨違規架構 - 10 大類別）...")

# ==================== 3. 核心融合邏輯 ====================
for ds_idx, ds_name in enumerate(DATASETS):
    if ds_idx >= len(DATASET_MAPPINGS):
        print(f"⚠️ 錯誤: 找不到資料集 {ds_name} 的類別對照表，已跳過。")
        continue

    mapping = DATASET_MAPPINGS[ds_idx]
    if not os.path.exists(ds_name):
        print(f"⚠️ 找不到資料夾: {ds_name}，將跳過此資料集。請檢查資料夾名稱！")
        continue

    print(f"📦 正在處理資料集: {ds_name}...")

    for split in SPLITS:
        possible_img_dirs = [
            os.path.join(ds_name, split, "images"),
            os.path.join(ds_name, "valid" if split == "val" else split, "images")
        ]
        possible_lbl_dirs = [
            os.path.join(ds_name, split, "labels"),
            os.path.join(ds_name, "valid" if split == "val" else split, "labels")
        ]

        img_src_dir = next((d for d in possible_img_dirs if os.path.exists(d)), None)
        lbl_src_dir = next((d for d in possible_lbl_dirs if os.path.exists(d)), None)

        if not img_src_dir or not lbl_src_dir:
            continue

        for img_name in os.listdir(img_src_dir):
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue

            base_name, ext = os.path.splitext(img_name)
            lbl_name = base_name + ".txt"

            new_base_name = f"ds{ds_idx}_{base_name}"
            new_img_name = new_base_name + ext
            new_lbl_name = new_base_name + ".txt"

            src_img_path = os.path.join(img_src_dir, img_name)
            dst_img_path = os.path.join(OUTPUT_DIR, split, "images", new_img_name)
            src_lbl_path = os.path.join(lbl_src_dir, lbl_name)
            dst_lbl_path = os.path.join(OUTPUT_DIR, split, "labels", new_lbl_name)

            shutil.copy2(src_img_path, dst_img_path)

            if os.path.exists(src_lbl_path):
                new_lines = []
                with open(src_lbl_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if not parts:
                            continue

                        try:
                            old_cls = int(parts[0])  # 確保正確抓取第 0 欄的類別
                        except ValueError:
                            continue

                        if old_cls in mapping:
                            new_cls = mapping[old_cls]
                            parts[0] = str(new_cls)
                            new_lines.append(" ".join(parts) + "\n")

                with open(dst_lbl_path, "w") as f:
                    f.writelines(new_lines)
            else:
                # 建立空白標籤（非常重要：前幾個資料集中沒有違規標籤的圖會在這裡變成完美的背景負樣本）
                open(dst_lbl_path, "w").close()


# ==================== 4. 自動生成全新 10 類 data.yaml ====================
def get_safe_path(dir_name, split_name):
    abs_path = os.path.abspath(os.path.join(dir_name, split_name, "images"))
    return abs_path.replace("\\", "/")


yaml_content = f"""train: {get_safe_path(OUTPUT_DIR, 'train')}
val: {get_safe_path(OUTPUT_DIR, 'val')}
test: {get_safe_path(OUTPUT_DIR, 'test')}

names:
  0: green
  1: red
  2: yellow
  3: car
  4: motor
  5: bike
  6: person
  7: truck
  8: solid-line
  9: red-light-violation
"""

with open(os.path.join(OUTPUT_DIR, "data.yaml"), "w") as f:
    f.write(yaml_content)

print("\n🎉 五合一 AI 直辨違規資料集融合完美完成！")
print(f"📂 新資料集路徑: {os.path.abspath(OUTPUT_DIR)}")
print(f"📝 訓練配置檔路徑: {os.path.abspath(os.path.join(OUTPUT_DIR, 'data.yaml'))}")
