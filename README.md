# Trip-Analyzer
This project aims to do the multi-used sentiment analysis of tourists' reviews via Text Mining and recommend tourist attractions to visitors with Word Cloud

The following five steps demostrate how to analyze the popular tourist spots from an amount of downloaded blog texts and generate a word cloud to display the result. Here, we take 46 tourist reviews as dataset examples and generate a word cloud with names of the most popular tourist attractions frequnetly mentioned in the reviews.

### Step 1: Import modules needed in making a wordcloud
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

## Stpe 2: Import spacy to do text mining
import spacy
nlp = spacy.cli.download("en_core_web_md")

# Step 3: Use spacy to pick out critical terminologies
  # Load the English NLP model
nlp = spacy.load('en_core_web_md')

  # create an dictionary to store the keywords
entity_dict_specified = {}

  # Read the 46 reveiws from Google Drive
for i in range(46):
  txt = open("/content/drive/My Drive/Colab Notebooks/kuala-lumpur-travel-blogs-dataset/" + str(i+1) + ".txt", "r").read()
  
  # Parse the text with spaCy
  doc = nlp(txt)
  for entity in doc.ents:
    text1 = entity.text 
    
  # Pick out terminologies relevant to locations, religious areas, buildings, etc        
    if entity.label_ == "LOC" or entity.label_ == "FAC" or entity.label_ == "GPE" or entity.label_ == "NORP" or entity.label_ == "PERSON":
      if text1 not in entity_dict_specified:
        entity_dict_specified[text1] = 1
      else:
        entity_dict_specified[text1] += 1
  # 'LOC': Non-GPE locations, mountain ranges, bodies of water
  # 'FAC': Buildings, airports, highways, bridges, etc.
  # 'GPE': Countries, cities, states
  # 'NORP': Nationalities or religious or political groups
  # 'PERSON': People, including fictional

s4 = [(k, entity_dict_specified[k]) for k in sorted(entity_dict_specified, key=entity_dict_specified.get, reverse=True)]

# Step 4: Store top 30 terms appearing in reviews most frequnetly into a list
KL_freq_most_common_list = [] 
for i in range(30):
  KL_freq_most_common_list.append(s4[i][0])

# Step 5: Generate a heart-shaped word cloud and store into Google Drive
import numpy as np
maskArray = np.array(Image.open("/content/drive/My Drive/Colab Notebooks/heart.png"))

def create_word_cloud(list1, maskarray):
  cloud = WordCloud(background_color = "white", max_words = 50, mask = maskarray, stopwords = STOPWORDS)
  t = ' '.join(list1)
  cloud.generate(t)
  plt.figure(figsize=(10,8),facecolor = 'white', edgecolor='blue')
  plt.imshow(cloud)
  plt.axis('off')
  plt.tight_layout(pad = 0)
  plt.show()
  cloud.to_file('/content/drive/My Drive/Colab Notebooks/wordcloud_by_Spacy.tif')
  

create_word_cloud (KL_freq_most_common_list, maskArray)
