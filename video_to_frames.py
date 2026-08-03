"""
這批跨越雙黃線的影片是非常寶貴的在地化資料！為了在 Android 手機上做到最精準、最不佔算力的「跨越雙黃線」科技執法，最頂級的作法是訓練一個 YOLOv8-Segmentation（實例分割）模型，或者是擴充原有的檢測模型。
不論你選擇哪一種，我們都必須把影片切成一張張的圖片，然後透過 Roboflow 進行精準標註。以下是完整的實戰教學：
------------------------------
## 🎬 步驟一：將雙黃線影片切成圖片（Python 腳本）
在標註之前，我們要先寫個簡單的腳本，每隔固定幀數（例如每秒抓 2-3 張）從影片中擷取圖片，避免重複度太高。請在專案中建立 video_to_frames.py 並執行：

import cv2import os
VIDEO_PATH = "double_yellow_line_video.mp4" # 🎬 你的雙黃線影片路徑OUTPUT_DIR = "yellow_line_frames"SAVE_INTERVAL = 15 # 每 15 幀抓一張圖（若影片是 30fps，等於每 0.5 秒抓一張）

os.makedirs(OUTPUT_DIR, exist_ok=True)cap = cv2.VideoCapture(VIDEO_PATH)frame_count = 0saved_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if frame_count % SAVE_INTERVAL == 0:
        cv2.imwrite(f"{OUTPUT_DIR}/frame_{saved_count:04d}.jpg", frame)
        saved_count += 1
    frame_count += 1

cap.release()
print(f"🎉 影片切圖完成！共匯出 {saved_count} 張圖片至 {OUTPUT_DIR} 資料夾。")

------------------------------
## 🎨 步驟二：選擇標註策略（二選一）
當你把圖片上傳到 Roboflow 後，針對「跨越雙黃線」，有以下兩種業界標準標註法。強烈建議選擇策略 A，因為它最適合你未來要做的科技執法。
## 💡 策略 A：實例分割標註法（最推薦 🌟）
我們不要用傳統的四方形方框，而是改用「多邊形套索工具（Polygon）」。

* 標註對象：
1. 將地上的「雙黃線」順著它的邊緣，用點對點的方式框出一個多邊形，類別命名為 double-yellow-line。
   2. 照常使用多邊形或方框，標註畫面中的 car 和 motor。
* 科技執法邏輯：訓練出來的模型會像螢光筆一樣，自動在手機畫面上把雙黃線的「精確像素範圍」塗滿。當車子的方框像素與雙黃線的螢光範圍產生交集時，App 就會自動判定跨越，完全不需要人工手動拉線！

## 💡 策略 B：關鍵邊界框標註法（傳統 YOLO 作法）
如果你不想用複雜的多邊形分割，想沿用之前的 8 大類別。

* 標註對象：
1. 遇到車輛「正在跨越」雙黃線的瞬間，把車輛本身框起來。
   2. 類別不要叫違規，直接歸類為 car 或 motor。
* 好處：這能強迫模型學習車輛在「側傾、壓線、逆向角度」時的特徵，因為一般公開資料集很少這種違規視角的照片。至於有沒有跨越，我們同樣可以在 Android 端用手動繪製「虛擬禁制區」來做幾何碰撞偵測。

------------------------------
## 🛠️ 步驟三：Roboflow 線上實作步驟

   1. 登入 Roboflow Universe，點擊 Create New Project。
   2. 專案類型選擇：
   * 如果你選策略 A，請選 Instance Segmentation（實例分割）。
      * 如果你選策略 B，請選 Object Detection（物件偵測）。
   3. 將剛剛切好的 yellow_line_frames 資料夾內所有圖片拖曳上傳。
   4. 開始標註：
   * 使用快捷鍵 P（多邊形工具）或 B（方框工具）。
      * 標註完所有圖片後，點擊 Add to Dataset，並按照 70% Train / 20% Valid / 10% Test 的比例切分。
   5. 點擊左側的 Generate 產生新版本的資料集。

------------------------------
## 🎯 接下來的整合策略
當你在 Roboflow 標註完這批特殊的雙黃線資料集後：

   1. 如果你選策略 B (Object Detection)：我們可以用之前寫的融合腳本，把這批新照片直接融入你原本的 6000 多張圖裡，直接啟動 300 輪的高精度訓練！
   2. 如果你選策略 A (Segmentation)：我們需要改寫訓練腳本，改跑 yolov8n-seg.pt 分割模型。

你覺得手上的影片量大約有幾部？你比較傾向使用免手動拉線的「策略 A（實例分割）」，還是延用現有架構的「策略 B（傳統方框）」呢？
"""

import cv2
import os

VIDEO_PATH = "double_white_line_video.mp4" # 🎬 你的雙黃線影片路徑
OUTPUT_DIR = "double_white_line_frames"
SAVE_INTERVAL = 15 # 每 15 幀抓一張圖（若影片是 30fps，等於每 0.5 秒抓一張）

os.makedirs(OUTPUT_DIR, exist_ok=True)
cap = cv2.VideoCapture(VIDEO_PATH)
frame_count = 0
saved_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if frame_count % SAVE_INTERVAL == 0:
        cv2.imwrite(f"{OUTPUT_DIR}/frame_{saved_count:04d}.jpg", frame)
        saved_count += 1
    frame_count += 1

cap.release()
print(f"🎉 影片切圖完成！共匯出 {saved_count} 張圖片至 {OUTPUT_DIR} 資料夾。")

