```python
import pandas as pd
import numpy as np
```


```python
true = pd.read_csv('True.csv')
```


```python
true.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>text</th>
      <th>subject</th>
      <th>date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>As U.S. budget fight looms, Republicans flip t...</td>
      <td>WASHINGTON (Reuters) - The head of a conservat...</td>
      <td>politicsNews</td>
      <td>December 31, 2017</td>
    </tr>
    <tr>
      <th>1</th>
      <td>U.S. military to accept transgender recruits o...</td>
      <td>WASHINGTON (Reuters) - Transgender people will...</td>
      <td>politicsNews</td>
      <td>December 29, 2017</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Senior U.S. Republican senator: 'Let Mr. Muell...</td>
      <td>WASHINGTON (Reuters) - The special counsel inv...</td>
      <td>politicsNews</td>
      <td>December 31, 2017</td>
    </tr>
    <tr>
      <th>3</th>
      <td>FBI Russia probe helped by Australian diplomat...</td>
      <td>WASHINGTON (Reuters) - Trump campaign adviser ...</td>
      <td>politicsNews</td>
      <td>December 30, 2017</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Trump wants Postal Service to charge 'much mor...</td>
      <td>SEATTLE/WASHINGTON (Reuters) - President Donal...</td>
      <td>politicsNews</td>
      <td>December 29, 2017</td>
    </tr>
  </tbody>
</table>
</div>




```python
fake = pd.read_csv('Fake.csv')
```


```python
fake.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>text</th>
      <th>subject</th>
      <th>date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Donald Trump Sends Out Embarrassing New Year’...</td>
      <td>Donald Trump just couldn t wish all Americans ...</td>
      <td>News</td>
      <td>December 31, 2017</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Drunk Bragging Trump Staffer Started Russian ...</td>
      <td>House Intelligence Committee Chairman Devin Nu...</td>
      <td>News</td>
      <td>December 31, 2017</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Sheriff David Clarke Becomes An Internet Joke...</td>
      <td>On Friday, it was revealed that former Milwauk...</td>
      <td>News</td>
      <td>December 30, 2017</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Trump Is So Obsessed He Even Has Obama’s Name...</td>
      <td>On Christmas day, Donald Trump announced that ...</td>
      <td>News</td>
      <td>December 29, 2017</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Pope Francis Just Called Out Donald Trump Dur...</td>
      <td>Pope Francis used his annual Christmas Day mes...</td>
      <td>News</td>
      <td>December 25, 2017</td>
    </tr>
  </tbody>
</table>
</div>




```python
true['label']=1
```


```python
fake['label'] = 0
```


```python
true.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>text</th>
      <th>subject</th>
      <th>date</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>As U.S. budget fight looms, Republicans flip t...</td>
      <td>WASHINGTON (Reuters) - The head of a conservat...</td>
      <td>politicsNews</td>
      <td>December 31, 2017</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>U.S. military to accept transgender recruits o...</td>
      <td>WASHINGTON (Reuters) - Transgender people will...</td>
      <td>politicsNews</td>
      <td>December 29, 2017</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Senior U.S. Republican senator: 'Let Mr. Muell...</td>
      <td>WASHINGTON (Reuters) - The special counsel inv...</td>
      <td>politicsNews</td>
      <td>December 31, 2017</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>FBI Russia probe helped by Australian diplomat...</td>
      <td>WASHINGTON (Reuters) - Trump campaign adviser ...</td>
      <td>politicsNews</td>
      <td>December 30, 2017</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Trump wants Postal Service to charge 'much mor...</td>
      <td>SEATTLE/WASHINGTON (Reuters) - President Donal...</td>
      <td>politicsNews</td>
      <td>December 29, 2017</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
news = pd.concat([fake, true],axis=0)
```


```python
news.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>text</th>
      <th>subject</th>
      <th>date</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Donald Trump Sends Out Embarrassing New Year’...</td>
      <td>Donald Trump just couldn t wish all Americans ...</td>
      <td>News</td>
      <td>December 31, 2017</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Drunk Bragging Trump Staffer Started Russian ...</td>
      <td>House Intelligence Committee Chairman Devin Nu...</td>
      <td>News</td>
      <td>December 31, 2017</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Sheriff David Clarke Becomes An Internet Joke...</td>
      <td>On Friday, it was revealed that former Milwauk...</td>
      <td>News</td>
      <td>December 30, 2017</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Trump Is So Obsessed He Even Has Obama’s Name...</td>
      <td>On Christmas day, Donald Trump announced that ...</td>
      <td>News</td>
      <td>December 29, 2017</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Pope Francis Just Called Out Donald Trump Dur...</td>
      <td>Pope Francis used his annual Christmas Day mes...</td>
      <td>News</td>
      <td>December 25, 2017</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
news.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>text</th>
      <th>subject</th>
      <th>date</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>21412</th>
      <td>'Fully committed' NATO backs new U.S. approach...</td>
      <td>BRUSSELS (Reuters) - NATO allies on Tuesday we...</td>
      <td>worldnews</td>
      <td>August 22, 2017</td>
      <td>1</td>
    </tr>
    <tr>
      <th>21413</th>
      <td>LexisNexis withdrew two products from Chinese ...</td>
      <td>LONDON (Reuters) - LexisNexis, a provider of l...</td>
      <td>worldnews</td>
      <td>August 22, 2017</td>
      <td>1</td>
    </tr>
    <tr>
      <th>21414</th>
      <td>Minsk cultural hub becomes haven from authorities</td>
      <td>MINSK (Reuters) - In the shadow of disused Sov...</td>
      <td>worldnews</td>
      <td>August 22, 2017</td>
      <td>1</td>
    </tr>
    <tr>
      <th>21415</th>
      <td>Vatican upbeat on possibility of Pope Francis ...</td>
      <td>MOSCOW (Reuters) - Vatican Secretary of State ...</td>
      <td>worldnews</td>
      <td>August 22, 2017</td>
      <td>1</td>
    </tr>
    <tr>
      <th>21416</th>
      <td>Indonesia to buy $1.14 billion worth of Russia...</td>
      <td>JAKARTA (Reuters) - Indonesia will buy 11 Sukh...</td>
      <td>worldnews</td>
      <td>August 22, 2017</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
news.isnull().sum()
```




    title      0
    text       0
    subject    0
    date       0
    label      0
    dtype: int64




```python
news = news.drop(['title', 'subject', 'date'], axis=1)
news.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Donald Trump just couldn t wish all Americans ...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>House Intelligence Committee Chairman Devin Nu...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>On Friday, it was revealed that former Milwauk...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>On Christmas day, Donald Trump announced that ...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Pope Francis used his annual Christmas Day mes...</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
news = news.sample(frac=1)     #Reshuffling
```


```python
news.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8391</th>
      <td>Democratic presidential candidate Martin O Mal...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18686</th>
      <td>DAR ES SALAAM (Reuters) - Tanzania shut down a...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3174</th>
      <td>WASHINGTON (Reuters) - California and other st...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13042</th>
      <td>BEIRUT (Reuters) - Iran s foreign ministry den...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12349</th>
      <td>The former CEO of a local cybersecurity firm i...</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
news.reset_index(inplace=True)
```


```python
news.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>text</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8391</td>
      <td>Democratic presidential candidate Martin O Mal...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>18686</td>
      <td>DAR ES SALAAM (Reuters) - Tanzania shut down a...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3174</td>
      <td>WASHINGTON (Reuters) - California and other st...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>13042</td>
      <td>BEIRUT (Reuters) - Iran s foreign ministry den...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12349</td>
      <td>The former CEO of a local cybersecurity firm i...</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
news.drop(['index'],axis=1, inplace = True)
```


```python
news.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Democratic presidential candidate Martin O Mal...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>DAR ES SALAAM (Reuters) - Tanzania shut down a...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>WASHINGTON (Reuters) - California and other st...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BEIRUT (Reuters) - Iran s foreign ministry den...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>The former CEO of a local cybersecurity firm i...</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
import re               #Regular expression module
```


```python
def wordopt(text):
    text = text.lower()                                  # for lower case conversion
    
    text = re.sub(r'https?://\S+|www\.\S+','',text)      # to remove urls, sub- substitute
    
    text = re.sub(r'<.*?>', '', text)                    # to remove HTML tags
    
    text = re.sub(r'[^\w\s]', '', text)                  # to remove punctuation
    
    text = re.sub(r'\d', '', text)                       # to remove digits
    
    text = re.sub(r'\n', ' ', text)                      # to remove newline characters
                  
    return text             
```


```python
news['text'] = news['text'].apply(wordopt)
```


```python
news['text']
```




    0        democratic presidential candidate martin o mal...
    1        dar es salaam reuters  tanzania shut down anot...
    2        washington reuters  california and other state...
    3        beirut reuters  iran s foreign ministry denied...
    4        the former ceo of a local cybersecurity firm i...
                                   ...                        
    44893    washington reuters  us democratic presidential...
    44894    a sick new  challenge  is going viral urging p...
    44895    donald trump increasingly desperate as the ele...
    44896    new york reuters  myanmar national security ad...
    44897     welcoming a prolife promarriage leader at the...
    Name: text, Length: 44898, dtype: object




```python
x = news['text']
y = news['label']
```


```python
x
```




    0        democratic presidential candidate martin o mal...
    1        dar es salaam reuters  tanzania shut down anot...
    2        washington reuters  california and other state...
    3        beirut reuters  iran s foreign ministry denied...
    4        the former ceo of a local cybersecurity firm i...
                                   ...                        
    44893    washington reuters  us democratic presidential...
    44894    a sick new  challenge  is going viral urging p...
    44895    donald trump increasingly desperate as the ele...
    44896    new york reuters  myanmar national security ad...
    44897     welcoming a prolife promarriage leader at the...
    Name: text, Length: 44898, dtype: object




```python
y
```




    0        0
    1        1
    2        1
    3        1
    4        0
            ..
    44893    1
    44894    0
    44895    0
    44896    1
    44897    0
    Name: label, Length: 44898, dtype: int64




```python
from sklearn.model_selection import train_test_split
```


```python
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.4)
```


```python
x_train.shape
```




    (26938,)




```python
x_test.shape
```




    (17960,)




```python
from sklearn.feature_extraction.text import TfidfVectorizer
```


```python
vectoriztion = TfidfVectorizer()
```


```python
xv_train = vectoriztion.fit_transform(x_train)

```


```python
xv_test = vectoriztion.transform(x_test)
```


```python
xv_train
```




    <26938x161386 sparse matrix of type '<class 'numpy.float64'>'
    	with 5543308 stored elements in Compressed Sparse Row format>




```python
xv_test
```




    <17960x161386 sparse matrix of type '<class 'numpy.float64'>'
    	with 3619390 stored elements in Compressed Sparse Row format>




```python
from sklearn.linear_model import LogisticRegression    #ML algorithem for classification
```


```python
LR = LogisticRegression()
```


```python
LR.fit(xv_train, y_train)
```




<style>#sk-container-id-1 {color: black;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LogisticRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">LogisticRegression</label><div class="sk-toggleable__content"><pre>LogisticRegression()</pre></div></div></div></div></div>




```python
pred_lr = LR.predict(xv_test)
```


```python
LR.score(xv_test, y_test)
```




    0.986358574610245




```python
print(classification_report(y_test, pred_lr))
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[43], line 1
    ----> 1 print(classification_report(y_test, pred_lr))


    NameError: name 'classification_report' is not defined



```python
from sklearn.tree import DecisionTreeClassifier     #ML algorithem for classification
```


```python
DTC = DecisionTreeClassifier()
```


```python
DTC.fit(xv_train, y_train)
```




<style>#sk-container-id-2 {color: black;}#sk-container-id-2 pre{padding: 0;}#sk-container-id-2 div.sk-toggleable {background-color: white;}#sk-container-id-2 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-2 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-2 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-2 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-2 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-2 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-2 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-2 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-2 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-2 div.sk-item {position: relative;z-index: 1;}#sk-container-id-2 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-2 div.sk-item::before, #sk-container-id-2 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-2 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-2 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-2 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-2 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-2 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-2 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-2 div.sk-label-container {text-align: center;}#sk-container-id-2 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-2 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-2" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>DecisionTreeClassifier()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" checked><label for="sk-estimator-id-2" class="sk-toggleable__label sk-toggleable__label-arrow">DecisionTreeClassifier</label><div class="sk-toggleable__content"><pre>DecisionTreeClassifier()</pre></div></div></div></div></div>




```python
pred_dtc = DTC.predict(xv_test)
```


```python
DTC.score(xv_test, y_test)
```




    0.9961024498886414




```python
from sklearn.metrics import classification_report

print(classification_report(y_test, pred_dtc))
```

                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00      9418
               1       1.00      1.00      1.00      8542
    
        accuracy                           1.00     17960
       macro avg       1.00      1.00      1.00     17960
    weighted avg       1.00      1.00      1.00     17960
    



```python
from sklearn.ensemble import RandomForestClassifier   #ML algorithem for classification
```


```python
rfc = RandomForestClassifier()
```


```python
rfc.fit(xv_train, y_train)
```




<style>#sk-container-id-5 {color: black;}#sk-container-id-5 pre{padding: 0;}#sk-container-id-5 div.sk-toggleable {background-color: white;}#sk-container-id-5 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-5 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-5 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-5 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-5 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-5 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-5 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-5 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-5 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-5 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-5 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-5 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-5 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-5 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-5 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-5 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-5 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-5 div.sk-item {position: relative;z-index: 1;}#sk-container-id-5 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-5 div.sk-item::before, #sk-container-id-5 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-5 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-5 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-5 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-5 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-5 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-5 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-5 div.sk-label-container {text-align: center;}#sk-container-id-5 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-5 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-5" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>RandomForestClassifier()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-5" type="checkbox" checked><label for="sk-estimator-id-5" class="sk-toggleable__label sk-toggleable__label-arrow">RandomForestClassifier</label><div class="sk-toggleable__content"><pre>RandomForestClassifier()</pre></div></div></div></div></div>




```python
predict_rfc = rfc.predict(xv_test)
```


```python
rfc.score(xv_test, y_test)
```




    0.9889198218262806




```python
print(classification_report(y_test, predict_rfc))
```

                  precision    recall  f1-score   support
    
               0       0.99      0.99      0.99      9418
               1       0.98      0.99      0.99      8542
    
        accuracy                           0.99     17960
       macro avg       0.99      0.99      0.99     17960
    weighted avg       0.99      0.99      0.99     17960
    



```python
from sklearn.ensemble import GradientBoostingClassifier
```


```python
gbc = GradientBoostingClassifier()
```


```python
gbc.fit(xv_train, y_train)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[7], line 1
    ----> 1 gbc.fit(xv_train, y_train)


    NameError: name 'xv_train' is not defined



```python
pred_gbc = gbc.predict(xv_test)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[8], line 1
    ----> 1 pred_gbc = gbc.predict(xv_test)


    NameError: name 'xv_test' is not defined



```python
gbc.score(xv_test, y_test)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[9], line 1
    ----> 1 gbc.score(xv_test, y_test)


    NameError: name 'xv_test' is not defined



```python
def output_label(n):
    if n==0:
        return "It is a Fake News"
    elif n == 1:
        return "It is Genuine News"
```


```python
def manual_testing(news):
    testing_news = {"test": [news]} 
    new_def_test = pd.DataFrame(testing_news)
    new_def_test['text'] = new_def_test['text'].apply(wordopt)
    new_x_test = new_def_test['text']
    new_xv_test = vectorization.transform(new_x_test)
    pred_lr = LR.predict(new_xv_test)
    pred_gbc = gb.predict(new_xv_test)
    pred_rfc = rfc.predict(new_xv_test)
    return "\n\nLR Prediction: {} \nGBC Prediction: {} \nRFC Prediction: {}".format(output_label(pred_lr[0]), ouput_label(pred_rfc[0]))
```


```python
news_article = str(input())
```


```python
manual_testing(news_article)
```


```python

```
