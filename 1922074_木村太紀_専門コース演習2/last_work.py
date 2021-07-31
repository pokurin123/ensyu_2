from flask import Flask, render_template, request, session
from datetime import datetime
import time
import cgi
import os, sys
import read_wakachi
import numpy as np
import pandas as pd
import glob
import re
import csv
import MeCab
import cv2
import cv2 as cv
import pickle
import matplotlib.pyplot as plt
from PIL import Image
import PIL.ExifTags as ExifTags
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import psycopg2

app = Flask(__name__)
#配置したDBよって変更
host = "ホスト名"
port = "ポート番号"
dbname = "db名"
user = "ユーザ名"
password = "パスワード"

conn = psycopg2.connect(\
    "host=" + host + \
    " port=" + port + \
    " dbname=" + dbname + \
    " user=" + user + \
    " password=" + password\
        )

#exifから位置情報
def get_gps(fname):
    # 画像ファイルを開く
    im = Image.open(fname)
    # EXIF情報を辞書型
    exif = {
        ExifTags.TAGS[k]: v
        for k, v in im._getexif().items()
        if k in ExifTags.TAGS
    }
    # GPS情報を得る
    gps_tags = exif["GPSInfo"]
    gps = {
        ExifTags.GPSTAGS.get(t, t): gps_tags[t]
        for t in gps_tags
    }
    # 緯度経度情報を得る
    def conv_deg(v):
        d = float(v[0])
        m = float(v[1])
        s = float(v[2])
        return d + (m / 60.0) + (s / 3600.0)
    lat = conv_deg(gps["GPSLatitude"])
    lat_ref = gps["GPSLatitudeRef"]
    if lat_ref != "N": lat = 0 - lat
    lon = conv_deg(gps["GPSLongitude"])
    lon_ref = gps["GPSLongitudeRef"]
    if lon_ref != "E": lon = 0 - lon
    return lat, lon

#分かち書き用
def get_sentense(text):
    tagger = MeCab.Tagger()
    tagger.parse("")
    node = tagger.parseToNode(text)
    docs = []
    result_dict_raw = {}
    wordclass_list = ['名詞','形容詞','動詞']
    not_fine_word_class_list = ["数", "非自立", "代名詞","接尾"]

    while node:
        word_feature = node.feature.split(",")
        word = node.surface

        word_class = word_feature[0]
        fine_word_class = word_feature[1]

        if ((word not in ['', ' ','\r', '\u3000']) \
            and (word_class in wordclass_list) \
            and (fine_word_class not in not_fine_word_class_list)):
            
            docs.append(word)
            result_dict_raw[word] = [word_class, fine_word_class]
        node = node.next
    return(docs)
#ベクトライズ
def vecs_array(documents):
    from sklearn.feature_extraction.text import TfidfVectorizer
 
    docs = np.array(documents)
    vectorizer = TfidfVectorizer(analyzer=get_sentense,binary=True,use_idf=False)
    vecs = vectorizer.fit_transform(docs)
    return vecs.toarray()

@app.route("/",methods=["POST","GET"])
def index():
    cur = conn.cursor()
    #マップ表示時
    if("map_id" in request.form):
        show_id = request.form["map_id"]
        cur.execute("SELECT * FROM last_work WHERE id = " + str(show_id))
        send_one = cur.fetchall()
        print("send_all",send_one)
        return render_template("last_work.html",send_one=send_one)

    #データアップロード後
    elif("file" in request.files):
        #データ書き込み-------------------------------------------------------------------
        item = request.files['file']
        if item:
            fout2 = open(os.path.join('./static/img_2', item.filename), 'wb')
            while True:
                chunk = item.read(1000000)
                if not chunk:
                    break
                fout2.write(chunk)
            fout2.close()

        img_path = './static/img_2/' + item.filename

        #インサート用id
        cur.execute("SELECT id FROM last_work")
        id = cur.fetchall()
        last_id = id[len(id)-1][0]
        last_id += 1
        #投稿画像の緯度経度
        lat, lon = get_gps(img_path)
        print(lat,lon)
        #投稿画像の年月日
        ymd = request.form["ymd"]
        #投稿画像の時間
        time = request.form["time"]
        #投稿画像のキーワード
        key_word = request.form["tag_word"]

        insert_txt = str(last_id) + ",'" + img_path + "'," + str(lat) + "," + str(lon) + ",'" + ymd + "','" + time + "','" + key_word + "'"

        cur.execute("INSERT INTO last_work VALUES("+ insert_txt +")")
        #ここまでデータ書き込み-------------------------------------------------------------------

        #データを持ってくる-------------------------------------------------------------------
        #ファイル名と類似度の辞書
        name_sim = {}

        cur.execute("SELECT * FROM last_work")
        nn = cur.fetchall()

        file_dbid = {}
        name_word = {}
        for i in nn:
            file_dbid[i[1]] = i[0]
            name_word[i[1]] = i[6]

        print(file_dbid)
        
        name_word_get = name_word[img_path]
        del name_word[img_path]
        #ここまでデータを持ってくる-------------------------------------------------------------------

        #画像の特徴点類似度-------------------------------------------------------------------
        #画像類似度
        TARGET_FILE = item.filename
        IMG_DIR = "./static/img_2/"
        IMG_SIZE = (200, 200)

        target_img_path = IMG_DIR + TARGET_FILE
        target_img = cv2.imread(target_img_path, cv2.IMREAD_GRAYSCALE)
        target_img = cv2.resize(target_img, IMG_SIZE)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        detector = cv2.AKAZE_create()
        (target_kp, target_des) = detector.detectAndCompute(target_img, None)

        print('TARGET_FILE: %s' % (TARGET_FILE))

        files = os.listdir(IMG_DIR)
        for file in files:
            if file == TARGET_FILE:
                continue
            comparing_img_path = IMG_DIR + file
            try:
                comparing_img = cv2.imread(comparing_img_path, cv2.IMREAD_GRAYSCALE)
                comparing_img = cv2.resize(comparing_img, IMG_SIZE)
                (comparing_kp, comparing_des) = detector.detectAndCompute(comparing_img, None)
                matches = bf.match(target_des, comparing_des)
                dist = [m.distance for m in matches]
                ret = sum(dist) / len(dist)
            except cv2.error:
                ret = 100000

            filename_zan = "./static/img_2/" + file
            name_sim[filename_zan] = ret
        print("画像辞書",name_sim)

        name_sim_2 = sorted(name_sim.items(), key=lambda x:x[1], reverse=True)
        name_score = {}
        name_score_count = 1
        name_score_mother = len(name_sim_2) + 1
        for i in name_sim_2:
            name_score[i[0]] = name_score_count / name_score_mother
            name_score_count += 1
        print("画像スコア",len(name_score))
        #ここまで画像の特徴点類似度-------------------------------------------------------------------

        #キーワードの類似度-------------------------------------------------------------------
        name_word_2 = {img_path:name_word_get}
        for i in name_word:
            name_word_2[i] = name_word[i]
        name_word_list = list(name_word_2.values())
        #print(name_word_2)

        cs_array = np.round(cosine_similarity(vecs_array(name_word_list), vecs_array(name_word_list)),5)
        print("コサイン",cs_array[0])
        name_cos_sim = {}
        del name_word_2[img_path]
        cos_count = 1
        for i in name_word_2:
            name_cos_sim[i] = cs_array[0][cos_count]
            cos_count += 1
        print("名前とコサイン",len(name_cos_sim))

        name_cos_score = {}
        name_cos_score_count = 1
        name_cos_mother = len(name_cos_sim) + 1
        for i in name_cos_sim:
            name_cos_score[i] = name_cos_score_count / name_cos_mother
            name_cos_score_count += 1
        print("ワードスコア",name_cos_score)
        #print(len(name_cos_score))
        #ここまでキーワードの類似度-------------------------------------------------------------------
        
        #総合スコア-------------------------------------------------------------------
        all_score = {}
        for i in name_cos_score:
            all_score[i] = ((name_cos_score[i]*1.2) + (name_score[i]*0.8)) /2
        print(all_score)

        all_score_2 = sorted(all_score.items(), key=lambda x:x[1], reverse=True)
        print("ソート後",all_score_2)
        score_id_count = 0
        send_id = []
        for i in all_score_2:
            if score_id_count <= 3:
                send_id.append(file_dbid[i[0]])
                score_id_count += 1
            else:
                break
        print("送るやつ",send_id)

        send_list = []
        for i in send_id:
            cur.execute("SELECT * FROM last_work WHERE id = " + str(i))
            zantei = cur.fetchall()
            send_list.append(zantei[0])
        print(send_list)

        conn.commit()
        return render_template("last_work.html",send_list=send_list)
    else:
        #トップページ
        return render_template("last_work.html")
