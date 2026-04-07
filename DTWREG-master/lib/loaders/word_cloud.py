import jieba
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# 标明文本路径，打开
text = open(r"D:\research\code\code_of_first_work\our_clip_work\DTWREG\DTWREG_raw_clip_reclip_location\DTWREG-master"
            r"\lib\loaders\word_cloud.txt", encoding="utf-8").read()
text = ' '.join(jieba.cut(text))
# 生成对象
wc = WordCloud(font_path="C:\Windows\Fonts\Microsoft YaHei UI\msyh.ttc", width=500, height=400, mode="RGBA",
               background_color=None).generate(text)
# 显示词云图
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.show()
# 保存文件
wc.to_file(r"D:\research\code\code_of_first_work\our_clip_work\DTWREG\DTWREG_raw_clip_reclip_location\DTWREG-master\lib\loaders\ciyun.png")
