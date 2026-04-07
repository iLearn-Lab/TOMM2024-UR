from wordcloud import WordCloud
import matplotlib.pyplot as plt

# 假设我们有以下单词列表
words = ["apple", "banana", "cherry", "date", "elderberry", "fig", "grape"]

# 由于所有单词出现的频率一样，我们可以为每个单词分配相同的频率
frequencies = {word: 1 for word in words}

# 创建WordCloud对象
wordcloud = WordCloud(width = 800, height = 800,
                background_color ='white',
                min_font_size = 10).generate_from_frequencies(frequencies)

# 绘制词云图
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)

plt.show()
