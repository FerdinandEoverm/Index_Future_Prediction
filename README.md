# Index_Future_Prediction
感谢您的关注！ <br>
这个库用于保存我个人对股指期货的相关研究，主要展示本人对传统统计、机器学习、以及深度学习相关的工具的研究和使用。 <br>

文件目录：<br>
framework：    保存了对整个研究体系外围框架的介绍说明，请先查看这个文件夹<br>
utils：        framework中提出的工具的python实现<br>
modules：      共用的底层模型组件，例如Self-attention、Embedding等<br>
statistic:     基础的统计学方法，大部分都outdate了，更多的是作为工具服务其他方法<br>
linear model： DLinear等线性模型，简单效果好，作为基线可以秒杀很多花里胡哨不切实际的论文<br>
RNN model：    RNN-based架构，主要是LSTM、GRU及其复合架构<br>
Temporal-only Transformer: 时间序列 Time Series Transformer，采用通道独立假设，将面板数据降维成普通时间序列，不建模截面关系，用于单资产预测<br>
Spatio-Temporal Transformer: 面板数据 Panel Data Transformer，采用通道相关假设，同时建模时间序列和截面关系，直接输出资产组合预测<br>


本人本科毕业于中国农业大学，研究生毕业于美国加州大学戴维斯分校。农业与资源经济学专业。主要研究微观经济学和计量经济学<br>
 <br>
CFA特许金融分析师持证人。曾任浙商期货产业研究所研究员。 <br>
目前未在职，希望能获取证券、期货的量化研究和交易相关的岗位。 <br>
<br>
虽然我的履历并不符合传统量化交易的路线，并非理工科出身，但我一直努力想要站到更广阔的平台。 <br>
2022年，通过CFA三级考试<br>
2023年，为了弥补数学知识的不足，我备考并参加了浙江大学数学系研究生考试，由于准备不充分未能通过，但已经充分复习了数学知识<br>
2024年，为了弥补计算机领域的不足，我备考并参加了浙江大学计算机系研究生考试，成功进入了复试，但也由于跨考的问题没能被录取<br>
2025年，由于已经补齐了相关基础，我开始动手实践量化相关问题并提高coding熟练度，这里就是存放我目前的学习和研究成果的地方<br>
如您对我感兴趣，欢迎随时联系我！ <br>
邮箱：ferdinandcici@outlook.com <br>
电话：+86 15637510406 <br>

Personal research on index future prediction. <br>
Apply multiple time series model, including statistical and deep-learning based, on one single task: predict and trade on index future market. <br>
Created by Chang Cao.<br>

Contact Information:<br>
E-mail：ferdinandcici@outlook.com<br>
Phone: +86 15637510406<br>
