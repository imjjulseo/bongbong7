import cv2
import numpy as np

def generate_fractal_noise(width, height, scale, octaves=4, persistence=0.5, lacunarity=2.0):
    """
    여러 크기의 랜덤 노이즈를 중첩하여 구름(연기) 형태의 텍스처를 생성합니다.
    """
    noise = np.zeros((height, width), dtype=np.float32)
    amplitude = 1.0
    frequency = scale
    total_amplitude = 0.0

    for _ in range(octaves):
        h_small = max(1, int(height / frequency))
        w_small = max(1, int(width / frequency))
        
        # 작은 해상도의 랜덤 노이즈 생성 후 부드럽게 확대 (구름 질감 형성)
        small_noise = np.random.rand(h_small, w_small).astype(np.float32)
        scaled_noise = cv2.resize(small_noise, (width, height), interpolation=cv2.INTER_CUBIC)
        
        noise += scaled_noise * amplitude
        total_amplitude += amplitude
        
        amplitude *= persistence
        frequency *= lacunarity

    return noise / total_amplitude

def add_smoke(image, center_x, center_y, radius=100, density=0.7, smoke_color=(200, 200, 200)):
    """
    지정된 좌표와 반경을 기준으로 이미지에 연기 효과를 합성합니다.
    
    - image: 합성할 원본 BGR 이미지 (배경)
    - center_x, center_y: 연기가 발생할 중심 픽셀 좌표
    - radius: 연기가 퍼지는 반경 (픽셀 단위)
    - density: 연기의 최대 짙기 (0.0 ~ 1.0)
    - smoke_color: 연기의 색상 (BGR 튜플, 기본값은 밝은 회색)
    """
    h, w = image.shape[:2]
    
    # 1. 화면 전체 크기의 구름 텍스처(노이즈) 생성
    # scale을 이미지 크기에 비례하게 주어 입자 크기를 조절
    noise = generate_fractal_noise(w, h, scale=max(w, h)/30.0, octaves=4)
    
    # 2. 지정된 좌표를 중심으로 하는 방사형 가우시안 마스크 생성
    Y, X = np.ogrid[:h, :w]
    dist_sq = (X - center_x)**2 + (Y - center_y)**2
    
    # 거리가 멀어질수록 자연스럽게 옅어지도록 설정 (sigma 제어)
    sigma = max(radius / 2.0, 1.0)
    mask = np.exp(-dist_sq / (2.0 * sigma**2))
    
    # 3. 노이즈 텍스처와 가우시안 마스크를 곱하여 최종 알파 맵(투명도) 산출
    smoke_alpha = noise * mask * density
    smoke_alpha = np.clip(smoke_alpha, 0.0, 1.0)
    smoke_alpha_3d = np.dstack([smoke_alpha] * 3)
    
    # 4. 알파 블렌딩 연산
    image_float = image.astype(np.float32)
    smoke_layer = np.full_like(image_float, smoke_color)
    
    blended = image_float * (1.0 - smoke_alpha_3d) + smoke_layer * smoke_alpha_3d
    return blended.astype(np.uint8)