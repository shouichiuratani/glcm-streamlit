import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray

# ページ設定
st.set_page_config(page_title="Potato GLCM (3D Surface, Blue-Yellow-Red)", layout="wide")
st.title("ポテト内部GLCMコントラスト: 3Dサーフェス (青=低, 黄=中, 赤=高)")

# 画像アップローダー
uploaded_files = st.file_uploader("画像をアップロード（複数可）", type=["png","jpg","jpeg"], accept_multiple_files=True)

# リサイズ寸法
RESIZE_DIM = (300, 300)

def preprocess_image_fixed_erosion(img, erosion_size=10):
    """
    1) 画像を300x300にリサイズ
    2) RGBA → マスク
    3) 外周10ピクセルエロージョンで境界削除
    4) 背景をポテト平均色で塗り潰す
    """
    img = img.resize(RESIZE_DIM, resample=Image.LANCZOS)
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    img_np = np.array(img)
    
    alpha_channel = img_np[:,:,3]
    mask = alpha_channel > 128
    
    kernel = np.ones((erosion_size, erosion_size), np.uint8)
    eroded_mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    
    # 背景をポテト平均色で埋める
    poteto_pixels = img_np[:,:,:3][mask]
    if len(poteto_pixels) == 0:
        mean_val = [0,0,0]
    else:
        mean_val = np.mean(poteto_pixels, axis=0)
    filled_img = np.tile(mean_val, (img_np.shape[0], img_np.shape[1], 1)).astype(np.uint8)
    filled_img[eroded_mask] = img_np[:,:,:3][eroded_mask]
    
    return filled_img, eroded_mask

def compute_glcm_contrast_grid(img_np, mask, grid_n=15, distance=5):
    """
    グリッド分割で distance=5, angles=[0,45,90,135]のGLCMコントラスト
    デフォルト grid_n=15
    """
    h, w, _ = img_np.shape
    cell_h = h // grid_n
    cell_w = w // grid_n
    
    gray = rgb2gray(img_np)
    glcm_contrast_grid = np.full((grid_n, grid_n), np.nan, dtype=np.float32)
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    
    for gy in range(grid_n):
        for gx in range(grid_n):
            y_start = gy*cell_h
            y_end   = min((gy+1)*cell_h, h)
            x_start = gx*cell_w
            x_end   = min((gx+1)*cell_w, w)
            
            sub_mask = mask[y_start:y_end, x_start:x_end]
            ratio = np.count_nonzero(sub_mask)/(sub_mask.size+1e-8)
            if ratio < 0.8:
                continue
            
            sub_gray = (gray[y_start:y_end, x_start:x_end]*255).astype(np.uint8)
            if sub_gray.shape[0]<2 or sub_gray.shape[1]<2:
                continue
            
            glcm = graycomatrix(sub_gray, distances=[distance], angles=angles,
                                levels=256, symmetric=True, normed=True)
            contrast = graycoprops(glcm, 'contrast').mean()
            glcm_contrast_grid[gy, gx] = contrast
    
    return glcm_contrast_grid

def create_custom_cmap():
    """
    青(低), 黄(中), 赤(高)を線形に補間したカラーマップを定義。
    0.0 -> 青, 0.5 -> 黄, 1.0 -> 赤
    """
    cdict = {
        'red':   [(0.0, 0.0, 0.0), (0.5, 1.0, 1.0), (1.0, 1.0, 1.0)],
        'green': [(0.0, 0.0, 0.0), (0.5, 1.0, 1.0), (1.0, 0.0, 0.0)],
        'blue':  [(0.0, 1.0, 1.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0)]
    }
    return LinearSegmentedColormap('BlueYellowRed', cdict)

def create_3d_surface(glcm_grid, cmap, original_img, alpha_surface, alpha_image):
    """
    GLCMコントラストを3Dサーフェスで可視化し、オリジナルのグレースケール画像をX,Y平面にオーバーレイする
    Z=コントラスト, X/Y=グリッドインデックス
    alpha_surface: GLCMサーフェスの透明度
    alpha_image: オリジナル画像の透明度
    """
    grid_ny, grid_nx = glcm_grid.shape
    X, Y = np.meshgrid(range(grid_nx), range(grid_ny))
    Z = np.nan_to_num(glcm_grid, nan=0.0)  # NaNは0埋め
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection='3d')
    
    # GLCMコントラストのサーフェスプロット
    surf = ax.plot_surface(X, Y, Z, cmap=cmap, edgecolor='none', alpha=alpha_surface)
    
    # オリジナル画像をグレースケールに変換
    grayscale = cv2.cvtColor(np.array(original_img), cv2.COLOR_RGB2GRAY)
    # グレースケール画像をRGBに戻す
    grayscale_rgb = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2RGB)
    # 画像をグリッドサイズにリサイズ
    img_resized = cv2.resize(grayscale_rgb, (grid_nx, grid_ny), interpolation=cv2.INTER_AREA)
    # 正規化
    img_normalized = img_resized / 255.0
    
    # X, Yのメッシュを作成
    X_img, Y_img = np.meshgrid(range(grid_nx), range(grid_ny))
    Z_img = np.zeros_like(X_img)  # Z=0の平面
    
    # 画像をテクスチャとしてプロット（透明度を調整）
    ax.plot_surface(X_img, Y_img, Z_img, rstride=1, cstride=1, facecolors=img_normalized, shade=False, alpha=alpha_image)
    
    # ラベルとタイトルの設定
    ax.set_xlabel("Grid X")
    ax.set_ylabel("Grid Y")
    ax.set_zlabel("Contrast")
    ax.set_title("3D GLCM Contrast Surface with Grayscale Image Overlay")
    
    # カラーバーの追加
    m = plt.cm.ScalarMappable(cmap=cmap)
    m.set_array(Z)
    fig.colorbar(m, ax=ax, shrink=0.5, aspect=10, label="Contrast")
    
    return fig

def overlay_heatmap_on_grayscale(filled_img, glcm_grid, custom_cmap, grid_n):
    """
    グレースケール画像の上にGLCMの2Dヒートマップをオーバーレイする
    """
    # グレースケール変換
    grayscale = cv2.cvtColor(filled_img, cv2.COLOR_RGB2GRAY)
    grayscale_rgb = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2RGB)
    
    # GLCMグリッドの正規化
    min_val = np.nanmin(glcm_grid)
    max_val = np.nanmax(glcm_grid)
    if max_val - min_val == 0:
        normalized = np.zeros_like(glcm_grid)
    else:
        normalized = (glcm_grid - min_val) / (max_val - min_val + 1e-8)
    normalized = np.nan_to_num(normalized)
    
    # カラーマップ適用
    heatmap = custom_cmap(normalized)
    heatmap = (heatmap[:, :, :3] * 255).astype(np.uint8)
    
    # ヒートマップを画像サイズにリサイズ
    heatmap_resized = cv2.resize(heatmap, (filled_img.shape[1], filled_img.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # オーバーレイ
    alpha = 0.5  # 透明度
    overlay = cv2.addWeighted(grayscale_rgb, 1 - alpha, heatmap_resized, alpha, 0)
    
    return overlay

if uploaded_files:
    st.sidebar.header("GLCMグリッド設定 (3D可視化)")
    grid_n = st.sidebar.slider("グリッド分割数 (2〜100)", 2, 100, 15)
    
    # 透明度調整スライダーの追加
    alpha_surface = st.sidebar.slider("3Dサーフェスの透明度 (0.0〜1.0)", 0.0, 1.0, 0.5, 0.05)
    alpha_image = st.sidebar.slider("オリジナル画像の透明度 (0.0〜1.0)", 0.0, 1.0, 0.8, 0.05)  # デフォルトを0.8に設定
    
    st.write(f"distance=5, 外周10px Erosion, grid={grid_n}, カラーマップ: Blue-Yellow-Red")
    
    custom_cmap = create_custom_cmap()  # カスタムカラーマップ (青→黄→赤)
    
    file_names = []
    all_grids = []
    overall_contrasts = []
    all_imgs = []
    all_overlays = []  # オーバーレイ画像を保存するリストを追加
    
    for file in uploaded_files:
        file_names.append(file.name)
        img = Image.open(file)
        
        filled_img, eroded_mask = preprocess_image_fixed_erosion(img, erosion_size=10)
        glcm_grid = compute_glcm_contrast_grid(filled_img, eroded_mask, grid_n=grid_n, distance=5)
        
        valid_vals = glcm_grid[~np.isnan(glcm_grid)]
        overall_contrast = float(np.mean(valid_vals)) if len(valid_vals) > 0 else 0.0
        
        all_grids.append(glcm_grid)
        overall_contrasts.append(overall_contrast)
        all_imgs.append(filled_img)
        
        # オーバーレイ画像の作成を追加
        overlay = overlay_heatmap_on_grayscale(filled_img, glcm_grid, custom_cmap, grid_n)
        all_overlays.append(overlay)
    
    for i, file_name in enumerate(file_names):
        st.subheader(f"ファイル: {file_name}")
        
        # 前処理された画像を表示
        st.image(all_imgs[i], caption="外周10pxエロージョン + 平均色埋め")
        
        # オーバーレイ画像を表示
        st.image(all_overlays[i], caption="グレースケール + GLCM 2D ヒートマップオーバーレイ")
        
        # GLCMコントラストの総合値を表示
        st.metric("GLCMコントラスト総合値 (平均)", f"{overall_contrasts[i]:.2f}")
        
        glcm_grid = all_grids[i]  # 正しいインデントに修正
        
        # 2Dヒートマップ表示
        fig_map, ax_map = plt.subplots()
        cax = ax_map.imshow(glcm_grid, cmap=custom_cmap)
        cbar = plt.colorbar(cax, ax=ax_map)
        cbar.set_label("Contrast")
        ax_map.set_title("GLCM Contrast Grid (2D Map)")
        st.pyplot(fig_map)
        
        # 3Dサーフェスプロットにオリジナル画像をオーバーレイ
        fig_3d = create_3d_surface(glcm_grid, custom_cmap, all_imgs[i], alpha_surface, alpha_image)
        st.pyplot(fig_3d)
    
    st.markdown("""
**3D可視化**  
カスタムカラーマップ青→黄→赤（線形補間）。  
青が低コントラスト、黄が中程度、赤が最も高いコントラストを示す。  
サイドバーのスライダーで3Dサーフェスとオリジナル画像の透明度を調整できます。  
3DサーフェスプロットのX,Y平面にグレースケール画像をオーバーレイしています。
""")

else:
    st.write("画像をアップロードしてください。")
