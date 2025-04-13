import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import cv2

# MNIST 데이터셋 로드
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 데이터 전처리
x_train = x_train / 255.0  # 정규화
x_test = x_test / 255.0    # 정규화
y_train = to_categorical(y_train, 10)  # One-hot encoding
y_test = to_categorical(y_test, 10)    # One-hot encoding

# 모델 로드 또는 학습
if os.path.exists("mnist_model.h5"):
    print("저장된 모델을 로드합니다...")
    model = tf.keras.models.load_model("mnist_model.h5")
else:
    print("모델을 학습하고 저장합니다...")
    # 모델 생성
    model = Sequential([
        Flatten(input_shape=(28, 28)),  # 28x28 이미지를 1차원으로 변환
        Dense(128, activation='relu'),  # 은닉층
        Dense(10, activation='softmax') # 출력층 (10개의 클래스)
    ])

    # 모델 컴파일
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # 모델 학습
    model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

    # 모델 저장
    model.save("mnist_model.h5")

# 모델 평가
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"테스트 정확도: {test_acc}")

# 모델 로드 함수
def load_trained_model():
    return tf.keras.models.load_model("mnist_model.h5")

# 학습된 모델 로드
model = load_trained_model()

# 모델 로드 후 재컴파일
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

def preprocess_and_predict_single(image_path):
    # 이미지 로드 및 이진화
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary_img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)

    # 노이즈 제거 (모폴로지 연산)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)

    # 윤곽선 감지
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 가장 큰 윤곽선 선택 (단일 숫자만 처리)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)

        # 숫자 영역 추출
        digit = binary_img[y:y+h, x:x+w]

        # 숫자를 중심에 정렬
        size = max(w, h)
        square_img = np.zeros((size, size), dtype=np.uint8)
        offset_x = (size - w) // 2
        offset_y = (size - h) // 2
        square_img[offset_y:offset_y+h, offset_x:offset_x+w] = digit

        # 크기 조정
        digit = cv2.resize(square_img, (28, 28))
        digit = digit / 255.0  # 정규화
        digit = np.expand_dims(digit, axis=-1)  # 채널 차원 추가
        digit = np.expand_dims(digit, axis=0)   # 배치 차원 추가

        # 예측 수행
        prediction = model.predict(digit)
        predicted_class = np.argmax(prediction)
        print(f"예측된 숫자: {predicted_class}")
    else:
        print("숫자를 감지하지 못했습니다.")

# 테스트 이미지 경로
image_path = "IMG_0169 2.JPG"  # 예제 이미지 경로
preprocess_and_predict_single(image_path)