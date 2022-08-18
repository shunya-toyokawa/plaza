import cv2
import pandas as pd
import numpy as np
from PIL import Image
BG_PATH    = "./inputs/image.jpg"   # 背景画像
ALPHA_PATH = "./inputs/alpha.png" # 合成アルファ画像
ALPHA_SCALE = 1.0

##合成する座標の算出
df = pd.read_csv('output.csv')
#print(df)
rshoulder_x =df["RShoulder_x"].values[0]
rshoulder_y =df["RShoulder_y"].values[0]
lshoulder_x = df["LShoulder_x"].values[0]
lshoulder_y = df["LShoulder_y"].values[0]


s= int((lshoulder_x - rshoulder_x) *0.7) #add_imgの横幅を決めている　左肩の座標と右肩の座標の差分に適当な倍率をかけたもの
x = int(rshoulder_x + (lshoulder_x -rshoulder_x)/6) #add_imgの座標決め
y= int(rshoulder_y)



# メイン関数
def main():
    add_img = cv2.imread(ALPHA_PATH, -1) # アルファチャンネルで読み込み
    pil_add_img= Image.fromarray(add_img)
    bg_img  = cv2.imread(BG_PATH)
    hi= int(pil_add_img.size[1] / (pil_add_img.size[0]/s))

    
    #resize
    add_img = cv2.resize(add_img,dsize=(s,hi))
    
    bg_img = merge_images(bg_img, add_img, x, y) # 座標を指定してアルファ画像を合成
    
    cv2.imwrite("./output/output.png", bg_img) # ファイル保存
    
    



# 画像を合成する関数(s_xは画像を貼り付けるx座標、s_yは画像を貼り付けるy座標)
def merge_images(bg, fg_alpha, s_x, s_y):
    alpha = fg_alpha[:,:,3]  # アルファチャンネルだけ抜き出す(要は2値のマスク画像)


    alpha = cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR) # grayをBGRに
    alpha = alpha / 255.0    # 0.0〜1.0の値に変換

    fg = fg_alpha[:,:,:3]

    f_h, f_w, _ = fg.shape # アルファ画像の高さと幅を取得
    b_h, b_w, _ = bg.shape # 背景画像の高さを幅を取得

    # 画像の大きさと開始座標を表示
    print("f_w:{} f_h:{} b_w:{} b_h:{} s({}, {})".format(f_w, f_h, b_w, b_h, s_x, s_y))

    bg[s_y:f_h+s_y, s_x:f_w+s_x] = (bg[s_y:f_h+s_y, s_x:f_w+s_x] * (1.0 - alpha)).astype('uint8') # アルファ以外の部分を黒で合成
    bg[s_y:f_h+s_y, s_x:f_w+s_x] = (bg[s_y:f_h+s_y, s_x:f_w+s_x] + (fg * alpha)).astype('uint8')  # 合成

    return bg

if __name__ == '__main__':
    main()