
data
--data_swedcad_mm SwedCAD和MM模型输入数据
--pre_data 数据预处理数据
----MMdata MM预处理数据
------3result 最终结果
------comet comet鉴定结果及其处理结果
------database 数据库
------mascot mascot鉴定结果及其处理结果
------mgf mgf源文件
------pFind pFind鉴定结果及其处理结果
----ProteomeToolsData ProteomeTools预处理数据
------InitialFile 初始文件
------raw_to_mgf 提取并处理后的mgf
------result 结果文件夹
data_preprocessing
--mm_data.py MM数据处理脚本
--proteometools_data.py ProteomeTools 数据处理脚本
improved-lstm-WGAN improved-WGAN和lstm结合模型
lstm LSTM模型
lstm-GAN lstm和GAN结合模型
seq2seq-WGAN seq2seq和WGAN结合模型


ProteomeTools数据处理详解：
1，初始源文件包含raw文件（InitialFile/raw/HCD/*.raw，例：01625b_GA1-TUM_first_pool_1_01_01-2xIT_2xHCD-1h-R1.raw）及其相应的鉴定结果（InitialFile/identification_results/HCD/*.zip，例：TUM_first_pool_1_01_01_2xIT_2xHCD-1h-R1-tryptic.zip）
2，利用SpectrumAnalyzer分析raw文件并将分析结果存放于相同目录（InitialFile/raw/HCD/*-analysis.txt，例：01625b_GA1-TUM_first_pool_1_01_01-2xIT_2xHCD-1h-R1-analysis.txt）
3，程序首先（内部过程省略）解压缩InitialFile/identification_results/HCD/下的所有压缩文件，提取每一个解压缩文件中的msms.txt，通过pif>=0.7及score>=100过滤msms.txt，同时在-analysis.txt中寻找相同SpectrumId的图谱的NCE，将其结果存放于InitialFile/identification_results/HCD/msms.txt。该msms.txt中包含Scan number,Raw File Name,Sequence,Charge,PEPMZ,PIF,Score,Modifications,NCE
4，通过pParse.exe（此操作是通过源码自动调用电脑中的pParse.exe程序）分析.raw得到相应的*_CIDIT.mgf,*_HCDFT.mgf和*__HCDIT.mgf文件（例：01625b_GA1-TUM_first_pool_1_01_01-2xIT_2xHCD-1h-R1_CIDIT.mgf等）。
5，遍历第四步中的.mgf文件，并结合第三步中的msms.txt写入新的mgf文件并存放于（raw_to_mgf/HCD/*.mgf，例01625b_GA1-TUM_first_pool_1_01_01-2xIT_2xHCD-1h-R1.mgf）
6，读取5中的所有mgf文件，分NCE分电荷存放结果于（result/mgf_result/HCD/NCE*/mgf文件名/charge*.txt）（例如：result/mgf_result/HCD/NCE20/mgf_01625b_GA1-TUM_first_pool_1_01_01-2xIT_2xHCD-1h-R1/charge2.txt）。该结果文件包含：NCE,SCORE,SpectrumName,Charge,SEQ,Modification,PepMass
7，读取6中的结果和5中相应的mgf文件匹配b,y,by,ay离子并将结果存放于（result/mgf_result/HCD/NCE*/mgf文件名/all_charge/match_ion/ay_by.txt或b_y.txt）

实验验证文件存放于 data/pre_data/ProteomeToolsData/例

其中：01625b_GA1-TUM_first_pool_1_01_01-2xIT_2xHCD-1h-R1.raw和TUM_first_pool_1_01_01_2xIT_2xHCD-1h-R1-tryptic.zip为原始文件
01625b_GA1-TUM_first_pool_1_01_01-2xIT_2xHCD-1h-R1.mgf为实验处理所得的mgf文件
NCE*_Charge2_*.txt为离子匹配的结果，此处列举了NCE20和NCE23的2价的ay,by,b,y的匹配结果。

