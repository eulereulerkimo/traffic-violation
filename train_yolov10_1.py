# https://share.google/aimode/z6e04FET2TiaZNs7B
"""
想要加入闖紅燈（Run a red light）與未依規定使用方向燈（No turn signal）這兩個非常實用的台灣常見違規場景，方向是完全正確的！
不過，從物件偵測（YOLO）的技術本質來看，這兩個行為屬於「動態連續動作（行為識別）」，無法單純靠看一張靜態圖片就判定有沒有闖紅燈或打燈。我們必須透過「物件偵測 + 邏輯判定」的複合式架構來達成。
以下是為您規劃如何將現有的 9 大類別擴充，並在實務上達成這兩項辨識的完整策略：
------------------------------
## 1. 類別定義與邏輯設計（如何辨識這兩個違規？）
我們不需要硬在 YOLO 裡面訓練一個叫做「闖紅燈」的框，而是要訓練模型認出關鍵元素，再用寫程式碼（Kotlin/Python）追蹤線條與位置來判定。
## A. 闖紅燈 (Run a red light)

* 核心邏輯：當【紅燈】亮起時，【車輛】的下邊緣（輪胎位置）跨越了【停止線/雙白實線】。
* 資料集需要擴充什麼？
* 您目前的類別已經有：1: red（紅燈）、3: car/4: motor（車輛）、8: solid-line（雙白實線/停止線）。
   * 💡 現有資料集已足夠支援！ 只要在追蹤演算法中，判定車輛邊框（Bounding Box）與停止線邊框在紅燈訊號下的相對座標重疊，即可觸發「闖紅燈」告警。

## B. 變換車道未打方向燈 (No turn signal)

* 核心邏輯：當【車輛】跨越【車道線（如白虛線或實線）】時，該車輛的【方向燈位置沒有閃爍】。
* 資料集需要擴充什麼？
* 您必須在資料集中，新增車道線與車輛方向燈（選用，難度極高）的標籤。
   * 💡 更務實的邊緣端作法：在手機端引入 ByteTrack 或 SORT 軌跡追蹤。當車輛的 X 軸座標發生橫向偏移（代表在變換車道），且偏移時車輛後方特定區域沒有亮度變化的特徵。
   * 如果想靠 YOLO 硬幹：您需要在資料集中加入 9: lane-line（車道線）。當車輛與車道線交會，就代表正在變換車道。

------------------------------
## 2. 升級版 10 大類別對齊表 (Data Mapping)
為了支援更精準的車道與違規判定，建議將資料集最終擴充為 10 大類別（加入一般的車道線/虛線）。
請在下一階段融合資料集時，將 data.yaml 與腳本類別修改如下：

names:
  0: green
  1: red
  2: yellow
  3: car
  4: motor
  5: bike
  6: person
  7: truck
  8: solid-line # 包含雙白線、停止線（判定闖紅燈、跨越雙白線）
  9: lane-line  # 新增：車道線/虛線（判定變換車道行為）

------------------------------
## 3. 如何實質提升「現有 9 類」的辨識準確度？
在您目前的 100 輪訓練完成後，如果發現紅綠燈太小看錯、或是雙白線漏判，請採取以下 YOLO 核心調優手段：
## ① 提高輸入解析度（最強效，但耗效能）
交通號誌在遠處時可能只有 10 × 10 像素。YOLOv10 預設用 imgsz=640 訓練，手機推論時很容易漏失。

* 作法：下一輪訓練將參數改為 imgsz=960 或 imgsz=1280。
* 代價：手機運算速度會變慢，但對於遠距離、小目標的準確度會有爆發性成長。

## ② 調整損失權重 (Loss Gain) 針對小目標
若發現紅綠燈（小目標）常常被車子（大目標）的 Loss 蓋掉，可以調整超參數中的 box 與 cls 權重，強迫模型更重視分類正確性。
## ③ 引入「物件追蹤 (Object Tracking)」避開單幀誤判
在 Android 實務部署時，絕對不能只靠單張圖片就判定。例如：

* 第 1 幀：沒看到紅燈（漏判）
* 第 2 幀：看到紅燈
* 第 3 幀：看到紅燈
* 如果引入 ByteTrack，對同一輛車與同一個紅燈進行跨幀的時間軸鎖定（例如連續 5 幀都有 3 幀以上抓到紅燈，才判定是紅燈），可以把靜態準確度從 85% 透過邏輯拉高到 98% 以上。

------------------------------
## 🚀 您接下來的下一步行動建議

   1. 讓目前的 100 輪跑完：先拿到現有的 best.pt，這份權重是您檢驗「車輛、紅綠燈、雙白線」基本辨識率的基準（Baseline）。
   2. 收集/標註新樣本：如果您有台灣本地馬路的錄影，可以截圖並用 LabelImg 或 Roboflow，專門針對「車道線（lane-line）」與「遠處微小紅綠燈」進行額外補充標註。
   3. 編寫後續的「違規判定邏輯」。

目前的訓練大約進展到第幾個 Epoch 了呢？等它跑完後，需要我先幫您準備如何在 Python 端寫出「車輛跨越停止線 + 紅燈 = 闖紅燈」的座標幾何判定演算法測試程式碼嗎？
"""
import os
import torch
from ultralytics import YOLO


def main():
    # ==================== 1. 環境檢查與路徑設定 ====================
    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ 目前使用的訓練設備: {'GPU (CUDA)' if device == '0' else 'CPU'}")

    yaml_path = os.path.abspath("merged_dataset/data.yaml")
    if not os.path.exists(yaml_path):
        print(f"❌ 錯誤：找不到配置檔 {yaml_path}，請先執行資料集融合腳本！")
        return

    # ==================== 2. 載入 YOLOv10-Nano 模型 ====================
    print("📥 正在下載/載入 YOLOv10n 預訓練權重...")
    model = YOLO("yolov10n.pt")

    # ==================== 3. 開始訓練模型 ====================
    print("🚀 開始訓練 YOLOv10 模型...")
    results = model.train(
        # --- 基本設定 ---
        data=yaml_path,
        epochs=100,             # 100輪（初次測試），後續視效果可拉長至200-300輪
        patience=25,            # 早停機制：25輪內沒進步就提早結束，避免過擬合
        imgsz=640,
        batch=16,               # RTX 2080 跑 Nano 模型設 16 非常安全，甚至可上調至 32
        device=device,
        workers=4,              # Windows 環境若出現 OOM 或凍結，請改為 0

        # --- 針對台灣交通場景與邊緣裝置的數據增強調優 ---
        mosaic=1.0,             # 100% 啟用 Mosaic，強迫模型學習小目標（紅綠燈）
        flipud=0.0,             # ❌ 嚴禁上下翻轉，避免紅綠燈與實線位置顛倒干擾學習
        fliplr=0.5,             # 允許左右翻轉，模擬對向車道或不同行車視角

        # --- 輸出路徑管理 ---
        project="traffic_yolov10",
        name="train_run",
        save=True,
        plots=True,             # 繪製 PR 曲線與 Loss 圖，方便評估訓練品質
    )
    print("🎉 模型訓練完成！")

    # ==================== 4. 導出 Android 專用格式 ====================
    print("🔄 嘗試將最佳模型權重導出為 Android (TFLite) 格式...")
    best_model_path = os.path.join("traffic_yolov10", "train_run", "weights", "best.pt")

    if os.path.exists(best_model_path):
        try:
            best_model = YOLO(best_model_path)
            # 這裡執行導出，若環境尚未修復會觸發 try-except 跳過，不影響前面辛苦訓練完的 pt 權重
            exported_path = best_model.export(format="tflite", int8=True)
            print(f"💾 Android 專用模型導出成功！")
            print(f"📍 TFLite 模型檔案路徑: {exported_path}")
        except Exception as e:
            print("\n⚠️ 轉檔失敗！這通常是因為 NumPy 環境損壞或尚未補裝 tensorflow-cpu 套件。")
            print("💡 請放心，你的訓練權重 'best.pt' 已安全儲存。")
            print("請手動執行對話開頭提供的【🛠️ 修復步驟】，再執行單獨轉檔腳本即可！\n")
    else:
        print("⚠️ 找不到最佳權重檔案，無法進行自動轉檔。")


if __name__ == "__main__":
    main()
