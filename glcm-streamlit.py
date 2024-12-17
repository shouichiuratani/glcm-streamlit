import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray

st.set_page_config(page_title="Potato GLCM (3D Surface, Blue-Yellow-Red)", layout="wide")
st.title("ポテト内部GLCMコントラスト: 3Dサーフェス (青=低, 黄=中, 赤=高)")

uploaded_files = st.file_uploader("画像をアップロード（複数可）", type=["png","jpg","jpeg"], accept_multiple_files=True)

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

def create_3d_surface(glcm_grid, cmap):
    """
    GLCMコントラストを3Dサーフェスで可視化
    Z=コントラスト, X/Y=グリッドインデックス
    """
    grid_ny, grid_nx = glcm_grid.shape
    X, Y = np.meshgrid(range(grid_nx), range(grid_ny))
    Z = np.nan_to_num(glcm_grid, nan=0.0)  # NaNは0埋め
    
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap=cmap, edgecolor='none')
    ax.set_xlabel("Grid X")
    ax.set_ylabel("Grid Y")
    ax.set_zlabel("Contrast")
    ax.set_title("3D GLCM Contrast Surface (Blue=Low,Yellow=Mid,Red=High)")
    fig.colorbar(surf, ax=ax, shrink=0.65, label="Contrast")
    return fig

if uploaded_files:
    st.sidebar.header("GLCMグリッド設定 (3D可視化)")
    grid_n = st.sidebar.slider("グリッド分割数(2~100)", 2, 100, 15)
    
    st.write(f"distance=5, 外周10px Erosion, grid={grid_n}, カラーマップ: Blue-Yellow-Red")
    
    custom_cmap = create_custom_cmap()  # カスタムカラーマップ (青->黄->赤)
    
    file_names = []
    all_grids = []
    overall_contrasts = []
    all_imgs = []
    
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
    
    for i, file_name in enumerate(file_names):
        st.subheader(f"ファイル: {file_name}")
        
        st.image(all_imgs[i], caption="外周10pxエロージョン + 平均色埋め")
        
        st.metric("GLCMコントラスト総合値(平均)", f"{overall_contrasts[i]:.2f}")
        
        glcm_grid = all_grids[i]
        
        # 2Dヒートマップ表示
        fig_map, ax_map = plt.subplots()
        cax = ax_map.imshow(glcm_grid, cmap=custom_cmap)
        cbar = plt.colorbar(cax, ax=ax_map)
        cbar.set_label("Contrast")
        ax_map.set_title("GLCM Contrast Grid (2D Map)")
        st.pyplot(fig_map)
        
        # 3Dサーフェス
        fig_3d = create_3d_surface(glcm_grid, custom_cmap)
        st.pyplot(fig_3d)
    
    st.markdown("""
**3D可視化**  
カスタムカラーマップ青→黄→赤（線形補間）。  
青が低コントラスト、黄が中程度、赤が最も高いコントラストを示す。
""")

else:
    st.write("画像をアップロードしてください。")

