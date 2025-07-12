import os
os.environ["ETS_TOOLKIT"] = "qt4"
os.environ["QT_API"] = "pyqt5"
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["PYVISTA_OFF_SCREEN"] = "true"

from mayavi import mlab
import numpy as np

# 配置Mayavi
mlab.options.offscreen = True

def create_and_save_plot():
    try:
        # 创建图形
        fig = mlab.figure(size=(1600, 900), bgcolor=(1, 1, 1))
        
        # 创建示例数据
        x, y, z = np.mgrid[-5:5:64j, -5:5:64j, -5:5:64j]
        values = np.sin(x*y*z)/(x*y*z)
        
        # 创建体绘制
        src = mlab.pipeline.scalar_field(values)
        mlab.pipeline.volume(src)
        
        # 保存图像
        mlab.savefig('volume_render.png')
        print("渲染成功保存为 volume_render.png")
    finally:
        # 确保关闭图形释放资源
        if mlab.gcf():
            mlab.close()

if __name__ == "__main__":
    create_and_save_plot()