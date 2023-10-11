# 世界幸福度分析项目

## 引言

### 摘要
随着物质生活水平的提高，人们开始关心主观精神生活水平。幸福度是客观物质生活水平和主观精神生活水平的综合指标。本项目使用2005-2021年的世界幸福度报告数据，利用Python中的数据科学库进行清洗、分析、可视化和建模，探究影响幸福度的因素，特别关注中国的幸福度状况。

### 研究背景

#### 世界幸福度报告简介
世界幸福度报告是联合国为衡量幸福的可持续发展方案而发布的国际调查报告。报告基于多方面因素进行研究，包括人均国内生产总值、健康预期寿命、社会支持等。报告的数据来自于全球范围的调查，具有广泛的参考价值。

#### 中国幸福度状况
本项目关注中国幸福度的变化情况，探讨影响中国幸福度的因素，以及如何提升中国的幸福度水平。

### 研究方法
本项目使用Python编程语言，利用NumPy、Pandas、Matplotlib、Plotly和Scikit-learn等库进行数据清洗、分析、可视化和建模。详细的数据介绍和处理方法请参见项目代码。

## 数据介绍

### 1 数据基本信息
项目使用的数据集包括2089条记录，涵盖了各国的幸福度以及多个相关指标。数据中存在缺失值，需要进行清洗和处理。
1.	可以看到该数据集共有2089条数据，每条数据由其Country name与year唯一标识。说明该数据集是由每个受调查国家每年的各项指标的数据组成的。
2.	数据的Country name和year未排序。
3.	数据中存在NaN的缺失数据。
### 2 原始数据一览


以下是从原始数据集中提取的样本数据：

| Country name                | Year | Life Ladder | Log GDP per capita | Social support | Healthy life expectancy at birth | Freedom to make life choices | Generosity | Perceptions of corruption | Positive affect | Negative affect | Confidence in national government |
|-----------------------------|------|-------------|--------------------|----------------|----------------------------------|-----------------------------|------------|---------------------------|----------------|----------------|----------------------------------|
| Haiti                       | 2006 | 3.754156    | 7.964181           | 0.693801       | 6.72                             | 0.449475                    | 0.361143   | 0.853506                  | 0.583196       | 0.332141       | 0.357021                         |
| Haiti                       | 2008 | 3.846329    | 8.004622           | 0.679098       | 17.360001                        | 0.464971                    | 0.219059   | 0.811659                  | 0.572523       | 0.255774       | 0.236633                         |
| Haiti                       | 2010 | 3.765999    | 7.97278            | 0.554031       | 28                               | 0.372941                    | 0.173648   | 0.848007                  | 0.495069       | 0.292557       | 0.156371                         |
| ...                         | ...  | ...         | ...                | ...            | ...                              | ...                         | ...        | ...                       | ...            | ...            | ...                              |
| Somaliland region           | 2012 | 5.057314    | NaN                | 0.786291       | NaN                              | 0.758219                    | NaN        | 0.333832                  | 0.686732       | 0.152428       | 0.651242                         |
| Hong Kong S.A.R. of China   | 2008 | 5.137262    | 10.815546          | 0.840222       | NaN                              | 0.922211                    | 0.293682   | 0.273945                  | 0.575073       | 0.236634       | 0.677437                         |
| Hong Kong S.A.R. of China   | 2009 | 5.397056    | 10.788493          | 0.834716       | NaN                              | 0.918026                    | 0.305077   | 0.272125                  | 0.606459       | 0.210104       | 0.619142                         |
| Hong Kong S.A.R. of China   | 2010 | 5.642835    | 10.846634          | 0.857314       | NaN                              | 0.890418                    | 0.32934    | 0.255775                  | 0.600561       | 0.183106       | 0.634737                         |
| Hong Kong S.A.R. of China   | 2011 | 5.474011    | 10.886932          | 0.84606        | NaN                              | 0.89433                     | 0.231902   | 0.244887                  | 0.582491       | 0.195712       | 0.584562                         |

总共包含 2089 行数据和 12 列指标。


**数据集指标解释**
- **Country name:** 国家名
- **Year:** 年份
- **Life Ladder:** 幸福度
- **Log GDP per capita:** 对数人均生产总值
- **Social support:** 社会支持
- **Healthy life expectancy at birth:** 预期健康寿命
- **Freedom to make life choices:** 人生抉择自由
- **Generosity:** 社会慷慨程度
- **Perceptions of corruption:** 社会清廉程度
- **Positive affect:** 积极影响
- **Negative affect:** 消极影响
- **Confidence in national government:** 对政府信任程度

**定义的DataFrame对象**
- **df_all_raw:** 原始数据
- **df_all_0:** 去重后的数据
- **df_all_1:** 排序后的数据
- **df_all_2:** 分国家补全Life Ladder及其后续列的数据
- **df_all_3:** 不分国家，用平均值补全Life Ladder及其后续列的数据
- **df_all_4:** 将Country name的字符串序列替换为[1,2...166]

### 3 数据清洗
#### 3.1 数据去重
在处理2089条数据时，我们首先进行了去重操作。每条数据由其Country name和year两个指标唯一标识。我们使用以下代码进行去重处理：
```
print("去重前")
print(df_all_raw.shape)
df_all_0 = df_all_raw.drop_duplicates(subset=['Country name', 'year'])
print("去重后")
print(df_all_0.shape)
df_all_0.to_csv("df_all_0.csv")
```

#### 3.2 数据排序
我们按照Country name进行升序排序，若Country name相同，则按照year进行升序排序。排序后的数据如下所示：


| Country name   | Year | Life Ladder | Log GDP per capita | Social support | Healthy life expectancy at birth | Freedom to make life choices | Generosity | Perceptions of corruption | Positive affect | Negative affect | Confidence in national government |
|-----------------|------|-------------|--------------------|-----------------|-----------------------------------|------------------------------|------------|---------------------------|-----------------|-----------------|------------------------------------|
| Afghanistan    | 2008 | 3.72359     | 7.302574           | 0.450662        | 50.5                              | 0.718114                     | 0.173169   | 0.881686                  | 0.414297        | 0.258195        | 0.612072                           |
| Afghanistan    | 2009 | 4.401778    | 7.472446           | 0.552308        | 50.799999                         | 0.678896                     | 0.195469   | 0.850035                  | 0.481421        | 0.237092        | 0.611545                           |
| Afghanistan    | 2010 | 4.758381    | 7.579183           | 0.539075        | 51.099998                         | 0.600127                     | 0.125859   | 0.706766                  | 0.516907        | 0.275324        | 0.299357                           |
| Afghanistan    | 2011 | 3.831719    | 7.552006           | 0.521104        | 51.400002                         | 0.495901                     | 0.167723   | 0.731109                  | 0.479835        | 0.267175        | 0.307386                           |
| Afghanistan    | 2012 | 3.782938    | 7.637953           | 0.520637        | 51.700001                         | 0.530935                     | 0.241247   | 0.77562                   | 0.613513        | 0.267919        | 0.43544                            |
| ...             | ...  | ...         | ...                | ...             | ...                               | ...                          | ...        | ...                       | ...             | ...             | ...                                |
| Zimbabwe       | 2017 | 3.6383      | 8.241609           | 0.754147        | 52.150002                         | 0.752826                     | -0.113937  | 0.751208                  | 0.733641        | 0.224051        | 0.682647                           |
| Zimbabwe       | 2018 | 3.61648     | 8.27462            | 0.775388        | 52.625                            | 0.762675                     | -0.084747  | 0.844209                  | 0.657524        | 0.211726        | 0.550508                           |
| Zimbabwe       | 2019 | 2.693523    | 8.196998           | 0.759162        | 53.099998                         | 0.631908                     | -0.08154   | 0.830652                  | 0.658434        | 0.235354        | 0.456455                           |
| Zimbabwe       | 2020 | 3.159802    | 8.117733           | 0.717243        | 53.575001                         | 0.643303                     | -0.029376  | 0.788523                  | 0.660658        | 0.345736        | 0.577302                           |
| Zimbabwe       | 2021 | 3.154578    | 8.153248           | 0.685151  

#### 3.3 数据补全
##### 3.3.1 部分数据缺失
对于部分数据缺失，我们使用每个国家该指标有数据的年份的数据平均值来补全没有数据的年份。


**一个国家一个国家地补全数据**
```
df_country_list = []
for countryname in countryname_list:
    df_country = df_all_1.loc[df_all_1['Country name'] == countryname]
    df_country.index = range(0, df_country.iloc[:,0].size)
    for factor in factors[2:]:
        df_country[factor] = df_country[factor].fillna(df_country[factor].mean())
    globals()[f"df_{countryname}"] = df_country
    df_country_list.append(df_country)
    df_country.to_csv(f"data_groupby_country/df_{countryname}.csv")

df_all_2 = pd.concat(df_country_list)
df_all_2.index = range(0, df_all_2.iloc[:,0].size)
df_all_2.to_csv("df_all_2.csv")
```
##### 3.3.2 全部数据缺失
对于全部数据缺失，我们使用全世界各国该指标的历年平均数据来补全该国该指标的数据。

```
df_all_3 = df_all_2.copy()
for factor in factors[2:]:
    df_all_3[factor] = df_all_3[factor].fillna(df_all_3[factor].mean())
df_all_3.index = range(0, df_all_3.iloc[:,0].size)
df_all_3.to_csv("df_all_3.csv")
```
#### 3.4 数据转换
在'2.1数据基本信息'分析中我们可以看到只有Country name的数据类型是object，而不是可进行比较的数字，故我们需要把Country name转化为可做比较的数字。
```
df_all_4=df_all_3.copy()
i=1
for countryname in countryname_list:
    df_all_4.loc[df_all_4['Country name']==countryname,'Country name']=i 
    i+=1
df_all_4.info() #得知Country name类型为object,需要转换
df_all_4['Country name']=df_all_4['Country name'].astype('int')
```
具体处理方法是将166个不同的Country name转化成1到166这166个整数。

### 4 数据分析及可视化
#### 4.1 各变量间相关性分析
##### 4.1.1 各指标两两间的散点图
```
factors1 = ['Life Ladder',
            'Country name',
            'year',
            'Log GDP per capita',
            'Social support',
            'Healthy life expectancy at birth',
            'Freedom to make life choices',
            'Generosity',
            'Perceptions of corruption',
            'Positive affect',
            'Negative affect']
sns.pairplot(df_all_4,vars=factors1,kind='reg',diag_kind='hist')
 ```
![](/image/pairplot1.png)
散点图中的点分布得越离散，表明这两个变量间的相关性越弱，越聚集在一条直线附近，表明这两个变量的线性相关性越强。
通过以上散点图，我们可以直观地看到哪些变量间的相关性更强，但是为了更好地量化变量间的相关性，我们需要用相关系数来准确地进一步衡量变量间的相关性。
##### 4.1.2 相关系数图
```
sns.heatmap(df_all_4.corr(), annot=True, cmap="YlGnBu");
```
![](/image/heatmap.png)\
(相关系数：0.8以上高度相关； 0.5-0.8 中度相关； 0.3-0.5 低度相关； 小于0.3 极弱，可视为不相关。)

可以看出：\
1.年份与除了自身外的因素的相关系数均小于0.3，可以认为年份对Life Ladder等指标几乎没影响，所以不需要分年份来进行分析\
2.Country name与除了自身外的因素均远远小于0.3，认为Country name对Life Ladder等指标没影响，所以不需要分国家来进行分析\
3.Life Ladder(幸福度)与Log GDP per capita,Social support,Healthy life expectancy at birth,Freedom to make life choices,Positive affect中度正相关，与Perceptions of corruption,Negative affect中度负相关，与year,Country name,Generosity,Confidence in national government几乎无关\
4.Life Ladder,Log GDP per capita,Social support,Healthy life expectancy at birth这几个指标两两间的相关性都很高

#### 4.2 建立多元线性回归模型
##### 4.2.1 去掉无关因素
由'4.1.1'的分析可知，Life Ladder与year,Country name,Generosity,Confidence in national government几乎无关，所以在建立预测Life Ladder的多元线性回归模型时可以不考虑这几个因素。
```
#回归模型
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

x = df_all_4[[column for column in ['Log GDP per capita',
                                    'Social support',
                                    'Healthy life expectancy at birth',
                                    'Freedom to make life choices',
                                    'Perceptions of corruption',
                                    'Positive affect',
                                    'Negative affect']]]
#既然Life Ladder(幸福度)与year,Country name,Generosity,Confidence in national government几乎#无关，那么预测模型里不用这几个参数
y = df_all_4['Life Ladder']
```
##### 4.2.2 确定训练集测试集数据量
虽说我们处于“大数据”时代，数据唾手可得，但是数据也是很珍贵的资源，对于数据工程来说快速生成一个比较准确的模型是主要任务，有的时候可能由于客观条件限制无法继续获得数据，这时候对于我们手中已有的数据，需要进行合理的划分：一部分用于训练模型（训练集）；另一部分用于测试模型（测试集）。
例如我们读取鸢尾花数据集（Scikit-Learn库中也自带该数据集），目标是根据花的四个特征确定是哪一种鸢尾花。那么每条数据包含四个浮点数和一个文字标签，我们把全部数据分为训练集和测试集，先使用训练集训练出模型，再将测试集的每一条数据输入模型，对比模型输出与理想输出是否一致。
小规模数据集（万数量级以下）训练集与测试集的划分比例一般为7:3或8:2；大规模数据集训练集与测试集的划分比例一般为98:2或99:1。
本实例中总数据量为2089条，数据量较小，故采用训练集:测试集=8:2。
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, shuffle=True)  # shuffle参数用于“洗牌”，打乱原始的数据行排列顺序
print('全体数据量：', len(x))
print('训练集数据量：', len(x_train))
print('测试集数据量：', len(x_test))


全体数据量： 2089
训练集数据量： 1671
测试集数据量： 418
##### 4.2.3 评估模型拟合效果
###### 4.2.3.1 均方误差进行评估
均方误差（Mean Squared Error, MSE）是一种常用的度量机器学习模型预测输出与实际输出之间差异的损失函数。它的计算方式是将模型的预测输出与实际输出之间的差的平方值进行求和，再将结果除以样本数量得到的平均值。
MSE的计算公式如下：
MSE = ∑(y_i - ŷ_i)^2 / n
其中，y_i表示实际输出，ŷ_i表示模型的预测输出，n表示样本数量。
MSE的值越小，说明模型的预测精度越高。因此，在训练机器学习模型时，通常会使用MSE作为损失函数，并不断优化模型的参数来尽可能减小MSE的值。
model=LinearRegression()
model.fit(x_train,y_train)
y_test_predict=model.predict(x_test)
mse=mean_squared_error(y_test,y_test_predict)
print("%0.2f"%mse)

0.29
然而，MSE的值本身并不能给出模型的精确程度，因为MSE值受数据范围的影响。例如，如果数据范围很大，即y_i和ŷ_i的差值可能很大，那么即使模型的预测精度很高，MSE的值也可能较大。因此，在评估模型精度时，通常需要将MSE值与数据范围进行比较，以确定MSE值是否较小。
4.2.3.2 判定系数进行评估
判定系数（Coefficient of Determination）是一种度量机器学习模型预测精度的指标。判定系数的值越接近1，表明模型的预测精度越高；判定系数的值越接近0，表明模型的预测精度越低。因此，判定系数可以用来衡量模型的预测精度，并为模型的选择和调优提供依据。
判定系数的计算公式如下：
R² = 1 - MSE / (∑(y_i - ȳ)^2 / n)
其中，MSE表示模型的均方误差，y_i表示实际输出，ȳ表示实际输出的平均值，n表示样本数量。
r2=r2_score(y_test,y_test_predict)
print(r2)

0.7700766885150759
该模型的判定系数接近0.8，可以看出模型预测效果较好。
4.2.4 Life Ladder预测模型
在检验了模型的准确度后，我们查看该模型的各个参数
model.coef_

array([ 0.37544935,  1.87352459,  0.02849311,  0.53045315, -0.60620074,
        2.24988852, -0.12408741])



model.intercept_

-2.744711029852029
最后得出的拟合公式为：Life Ladder=0.37544935*Log GDP per capita+1.87352459*Social support+0.02849311*Healthy life expectancy at birth+0.53045315*Freedom to make life choices-0.60620074*Perceptions of corruption+2.24988852*Positive affect-0.12408741*Negative affect-2.744711029852029

#### 4.3 世界幸福度地图
此处我们画出2021年的世界幸福度地图，有助于我们比较2021年世界不同地区的幸福度差异，可以直观地看出世界各地区的幸福度水平。
首先需要从总的数据中提取2021年的数据：
我们先得到2005-2021的分年份的dataframe，即df_2005,df_2006,...,df_2021。
dfyear_list = []
for i in range(2005, 2022):
    dfyear_list.append('df_'+ str(i))
for i in range(2005, 2022):
    #exec函数能执行函数中" "内的操作
    exec("%s = df_all_3.loc[df_all_3['year']==%d]" %(dfyear_list[i-2005],i))
    exec("%s.index=range(0,%s.iloc[:,0].size)"%(dfyear_list[i-2005],dfyear_list[i-2005]))
    exec("%s.to_csv('data_groupby_year/%s')"%(dfyear_list[i-2005], 'df_'+str(i)+'.csv'))
这样以后即可得到每年的数据。

再用df_2021来绘制2021年的世界幸福度地图(也可用其他年份的数据来绘制其他年份对应的世界幸福度地图)：
data = dict(type = 'choropleth', 
           locations = df_2021['Country name'],
           locationmode = 'country names',
           colorscale = 'RdYlGn',
           z = df_2021['Life Ladder'], 
           text = df_2021['Country name'],
           colorbar = {'title':'happiness'})

layout = dict(title = 'Geographical Visualization of Happiness Score in 2021', 
              geo = dict(showframe = False))

choromap3 = go.Figure(data = [data], layout=layout)
plot(choromap3, filename='画图/世界幸福度地图.html')
 

分析这张2021世界幸福度地图，我们可以看到：
	可以看出北欧的幸福度最高，欧洲，北美洲，大洋洲的幸福度普遍很高。结合这些地区状况，我们可以推断其幸福度很高可能与其良好的社会福利与社会保障，较高的人均收入水平相关。
	西亚尤其是阿富汗、南亚，非洲地区的幸福度普遍很低。结合地理知识，这些可能与其落后的经济水平，社会各方面发展水平相关，而西亚部分地区还存在战乱，这可能也是影响其幸福度的重要因素。
	西亚的阿拉伯半岛地区，非洲的好望角地区幸福度比其周围普遍偏低的幸福度高出不少，因为西亚的阿拉伯地区盛产石油，而非洲的好望角地区是重要的航道结点。
	中亚，东亚，北亚，东南亚，以及南美洲幸福度良好。
4.4 幸福度排名靠前与靠后国家
4.4.1 幸福度排名最靠前国家
4.4.1.1 2021年幸福度排名前十国家
首先列出2021年世界幸福度排名前10的国家
df_2021_sorted=df_2021.sort_values(by='Life Ladder',ascending=False)
top10=df_2021_sorted.head(10)[['Country name','Life Ladder']] 
top10.index=range(1,top10.iloc[:,0].size+1)

fig = px.bar(top10, 
             x="Country name", 
             y="Life Ladder", 
             color="Country name", 
             title="World's happiest countries in 2021")
fig.show()
 

可以看出，2021年世界幸福度排名前十的国家依次是：芬兰，丹麦，以色列，冰岛，瑞典，挪威，瑞士，荷兰，新西兰，澳大利亚。
4.4.1.2 世界幸福度排名第一的国家
下面我们来分析世界上最幸福的国家是哪个国家
列出2005-2021历年最幸福的国家：
top_country_list = []
for i in range(2005, 2022):
    # exec函数能执行函数中" "内的操作
    exec("top_country=%s.sort_values(by=['Life Ladder'],ascending=False)"%dfyear_list[i-2005])
    exec("top_country=top_country.iloc[0,0:3]")
    top_country_list.append(top_country)

df_topcountry = pd.DataFrame(top_country_list)
df_topcountry.index = range(0, df_topcountry.iloc[:,0].size)
df_topcountry
	Country name	year	Life Ladder
0	Denmark	2005	8.018934
1	Finland	2006	7.672449
2	Denmark	2007	7.834233
3	Denmark	2008	7.970892
4	Denmark	2009	7.683359
5	Denmark	2010	7.770515
6	Denmark	2011	7.788232
7	Switzerland	2012	7.776209
8	Canada	2013	7.593794
9	Denmark	2014	7.507559
10	Norway	2015	7.603434
11	Finland	2016	7.659843
12	Finland	2017	7.788252
13	Finland	2018	7.858107
14	Finland	2019	7.780348
15	Finland	2020	7.88935
16	Finland	2021	7.794378
可以看出Finland与Denmark各当过7次第一,为了进一步比较芬兰与丹麦谁更幸福，我们比较其2005-2021年的平均幸福度：
世界各国的幸福度平均值从高到低排列：
Country name
Denmark                     7.681457
Finland                     7.611299
Switzerland                 7.528177
Norway                      7.498840
Iceland                     7.459697
                              ...   
Togo                        3.603208
Burundi                     3.548124
Central African Republic    3.514954
Afghanistan                 3.505506
South Sudan                 3.401875
Name: Life Ladder, Length: 166, dtype: float64
可以看出，2005-2021年平均幸福度最高的是丹麦。
综上，我们可以认为丹麦是2005-2021年最幸福的国家。
4.4.2 幸福度排名最靠后国家
4.4.2.1 2021幸福度排名后十位的国家
df_2021_sorted=df_2021.sort_values(by='Life Ladder',ascending=True)
btm10=df_2021_sorted.head(10)[['Country name','Life Ladder']] #btm是bottom的缩写 
btm10.index=range(1,top10.iloc[:,0].size+1)   
btm10
	Country name	Life Ladder
1	Lebanon	2.178809
2	Afghanistan	2.436034
3	Zambia	3.082155
4	Zimbabwe	3.154578
5	India	3.558254
6	Malawi	3.635283
7	Tanzania	3.680568
8	Sierra Leone	3.714294
9	Jordan	3.909149
10	Egypt	4.025748
fig = px.bar(btm10, 
             x="Country name", 
             y="Life Ladder", 
             color="Country name", 
             title="World's least happiest countries in 2021")
fig.show()
 

2021幸福度排名后十位的国家依次是黎巴嫩、阿富汗、赞比亚、津巴布韦、印度、马拉维、坦桑尼亚、塞拉利昂、约旦、埃及。
4.4.2.2 世界幸福度排名倒数第一的国家
btm_country_list = []
for i in range(2005, 2022):
    # exec函数能执行函数中" "内的操作
    exec("btm_country=%s.sort_values(by=['Life Ladder'],ascending=True)"%dfyear_list[i-2005])
    exec("btm_country=btm_country.iloc[0,0:3]")
    btm_country_list.append(btm_country)

df_btmcountry = pd.DataFrame(btm_country_list)
df_btmcountry.index = range(0, df_btmcountry.iloc[:,0].size)
df_btmcountry
	Country name	year	Life Ladder
0	Turkey	2005	4.718734
1	Togo	2006	3.202429
2	Zimbabwe	2007	3.280247
3	Togo	2008	2.807855
4	Tanzania	2009	3.407508
5	Tanzania	2010	3.229129
6	Togo	2011	2.936221
7	Syria	2012	3.164491
8	Syria	2013	2.687553
9	Togo	2014	2.838959
10	Liberia	2015	2.701591
11	Central African Republic	2016	2.693061
12	Afghanistan	2017	2.661718
13	Afghanistan	2018	2.694303
14	Afghanistan	2019	2.375092
15	Lebanon	2020	2.633753
16	Lebanon	2021	2.178809
可以看出Togo一共排过4次倒数第一。
4.5 分析中国幸福度数据
4.5.1 中国幸福度世界排名
首先获得从总的数据集中获取中国的数据，并且将数据的index设置为中国的历年幸福度排名：
China_rank_list=[]
for i in range(2005, 2022):
    # exec函数能执行函数中" "内的操作
    exec("dfbyyear_sort=%s.sort_values(by=['Life Ladder'],ascending=False)"%dfyear_list[i-2005])
    dfbyyear_sort.index=range(1,dfbyyear_sort.iloc[:,0].size+1)
    exec("series_china=dfbyyear_sort.loc[dfbyyear_sort['Country name']=='China']")
    China_rank_list.append(series_china)

df_China_rank = pd.concat(China_rank_list)
df_China_rank
然后画出中国历年排名折线图：
plt.rcParams['font.sans-serif'] = ['SimHei']  # 添加这条可以让图形显示中文

x_axis_data = df_China_rank['year']
y_axis_data = df_China_rank.index

# plot中参数的含义分别是横轴值，纵轴值，线条格式，颜色，透明度,线的宽度和标签
plt.plot(x_axis_data, y_axis_data, 'ro-', color='blue',
         alpha=1, linewidth=1, label='China happiness ranking')
 
# 给这个折线图中的点加数据标签
for x, y in zip(x_axis_data, y_axis_data):
    plt.text(x, y+0.3, '%.0f'%y, ha='center', va='bottom', fontsize=12)
    # Add the text *s* to the Axes at location *x*, *y*(x,y)
 
# 显示标签，如果不加这句，即使加了label='一些数字'的参数，最终还是不会显示标签
plt.legend(loc="best")
plt.xlabel('年份')
plt.ylabel('中国幸福度世界排名')
plt.title("2005-2021中国幸福度世界排名", fontdict={'size': 25})
plt.show()
 

可以看出中国的世界幸福度排名在58名到94名的区间内，而2020年幸福度排名忽然从91名上升到了58名,这或许与中国在疫情防控方面取得的成就有关。
4.5.2 中国怎样提升幸福度
了解到了中国的幸福度世界排名后，我们自然而然地想到一个问题，即中国最容易从哪个方面下手来提升幸福度。
为此，我们先分析中国各项指标与世界平均水平的差异(因为单单分析中国的数据，数据量较少，为了保障比较的客观准确，我们这里采用未经过补全的原始数据)：
world_mean=df_all_1.mean()
China_mean=df_China_raw.mean()
gap=China_mean-world_mean 
print("中国各项指标平均值-世界平均水平：")
print(gap)


中国各项指标平均值-世界平均水平：
year                                -0.227621
Life Ladder                         -0.374812
Log GDP per capita                  -0.070427
Social support                      -0.022383
Healthy life expectancy at birth     4.252487
Freedom to make life choices         0.102936
Generosity                          -0.158677
Perceptions of corruption                 NaN
Positive affect                      0.039517
Negative affect                     -0.102264
Confidence in national government         NaN
dtype: float64
可以看出：
1.中国在2005-2021年的平均幸福度低于世界平均水平
2.做得好的方面：中国的Healthy life expectancy at birth,Freedom to make life choices,Positive affect高于平均水平，Negative affect低于世界平均水平(注意Negative affect与幸福度是负相关的)
3.有待提升方面：中国的Log GDP per capita,Social support,Generosity低于世界平均水平

为了分析中国最容易从哪方面下手提升总体幸福度，这里我们提出一下朴素的假设与简化的处理方法：
	假设①:提升低于平均水平的方面比提升高于平均水平的方面更容易
	-->简化处理方法:
	只考虑低于平均水平方面
	-->具体操作:
	只考虑中国的低于世界平均水平指标:Log GDP per capita, Social support,Generosity 
	假设②:低于平均水平越大，提升空间越大
	-->简化处理方法:
	将与平均水平的差值的绝对值记为提升潜力
	-->具体操作:
	提升潜力
	Log GDP per capita:0.070427
	Social support:0.022383
	Generosity:0.158677
	假设③:某指标提升潜力值越大，在幸福度多元线性模型中权重越高，从该指标下手越容易提升幸福度
	-->简化处理方法:
	将每个指标的提升潜力*对应权重记作加权潜力，找出加权潜力最大的那个指标，则从该指标下手最容易提升整体幸福度
	-->具体操作:
	加权潜力
	Log GDP per capita:0.02644177137245
	Social support:0.04193510089797
	Generosity:0(因为线性模型里没考虑这个因素，其权重为0)
	综上分析，中国最容易通过提升Social support来提升整体幸福度。


## 使用方法

1. 下载项目代码和数据集。
2. 在Python环境中运行 `analysis.ipynb` Jupyter Notebook 文件，按照代码步骤进行数据分析。

## 参考
- [世界幸福度报告官网](https://worldhappiness.report/)

欢迎查看和贡献代码！如果您有任何问题或建议，请在Issues中提出。
