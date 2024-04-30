import requests
from flask import Flask, request, jsonify, render_template, redirect,Markup
from torchvision import transforms
from PIL import Image
import torch
import pickle
import numpy as np
from utils.disease import disease_dic

app = Flask(__name__)

# 加载模型
disease_classes = ['根结线虫病', '根肿病', '根腐病', '灰霉病', '烟煤病', '病毒病', '白粉病', '脐腐病', '软腐病', '锈病', '青枯病']
model = torch.load('Models/plant_disease11_model.pth', map_location=torch.device('cpu'))
model.eval()

# 图像预处理
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
@app.route('/disease-detection/', methods=['GET', 'POST'])
def disease_detection():
    if request.method == 'POST':
        # 处理 POST 请求，执行疾病检测操作
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        try:
            image = Image.open(file.stream)
            # 如果图片是 4 通道的（RGBA 格式），转换为 3 通道的 RGB 格式
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            # 图像预处理
            image_tensor = preprocess(image).unsqueeze(0)
            with torch.no_grad():
                output = model(image_tensor)
            predicted_class = torch.argmax(output).item()
            # 获取预测的疾病名称
            predicted_disease = disease_classes[predicted_class]
            predicted_disease = Markup(str(disease_dic[predicted_disease]))
            print(predicted_disease)
            # 返回预测结果
            return render_template('disease-result.html', predicted_disease=predicted_disease)
        except:
            pass
            print("Exception")
        return render_template('disease.html')
    else:
        # 处理 GET 请求，渲染表单页面
        return render_template('disease.html')



def weather_fetch(city_name):
    api_key = "6ee1dc7bb1bc5a8a16ba813050149c05"
    base_url = "http://api.openweathermap.org/data/2.5/weather?"

    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    response = requests.get(complete_url)
    x = response.json()

    if x["cod"] != "404":
        y = x["main"]

        temperature = round((y["temp"] - 273.15), 2)
        humidity = y["humidity"]
        return temperature, humidity
    else:
        return None


crop_recommendation_model = pickle.load(
    open('Models/model.pkl', 'rb'))


@app.route('/crop-planning/')
def crop_planning():
    return render_template("crop.html")


@ app.route('/crop-predict/', methods=['POST'])
def crop_prediction():
    title = 'Kisan++ - Crop Recommendation'

    if request.method == 'POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['potassium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        city = request.form.get("city")

        if weather_fetch(city) != None:
            temperature, humidity = weather_fetch(city)
            data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            my_prediction = crop_recommendation_model.predict(data)
            final_prediction = my_prediction[0]
            print()
            print("Final Prediction", final_prediction)
            return render_template('crop-result.html', prediction=final_prediction.capitalize(), title=title)
        else:
            return render_template('try_again.html', title=title)

@app.route('/sell/')
def sell():
    return render_template("sell.html")

@app.route('/buy/')
def buy():
    return render_template("buy.html")

@app.route('/')
def home():
    return render_template('index.html')

@app.errorhandler(404)
def not_found(e):
    return render_template("404.html")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)


